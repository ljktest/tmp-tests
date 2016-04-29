!USEFORTEST toolbox
!USEFORTEST avgcond
!USEFORTEST postprocess
!USEFORTEST advec
!USEFORTEST interpolation
!USEFORTEST io
!USEFORTEST topo
!> @addtogroup cart_structure
!! @{

!------------------------------------------------------------------------------

!
! MODULE: cart_topology
!
!
! DESCRIPTION:
!>  This module provide a cartesien topology on the parrallel layout.
!
!> @details
!!  This module provide a cartesien topology on the parrallel layout.
!! This virtual topology is created by the MPI procedures (and thus use
!! low-level optimisation based on the underlyinfg hardware). It
!! provides the different tools to create, to manipulate and to interface
!! it with the other topology and communicators.
!! The solver use some dimensionnal splitting and this module contains all the
!! method used to solve advection along the Y-axis. This is a parallel
!! implementation using MPI and the cartesien topology it provides.
!!
!!  Nowaday, the domain is only splitted along Y and Z axis. Therefore,
!! we only use a 2D cartesian topology.
!!  A "global" communicator is devoted to the (2D) cartesian structure.
!! Another communicator is added for each direction in order to deal
!! with all 1D communication (along Y or Z).
!! Be careful : the (Y,Z)-axis in the 3D mesh match to the (X,Y) axis on the 2D
!! mpi-topology.
!
!> @author
!! Jean-Baptiste Lagaert, LEGI
!
!------------------------------------------------------------------------------

module cart_topology

    use precision_tools
    use cart_mesh_tools
    use mpi, only: MPI_TAG_UB

    implicit none

    ! ===== Structure =====
    ! ----- Structure to save work item information -----
    ! This allow to use different resolution more easily.
    type group_info
      !> Computation are done by group of line. Here we define their size
      integer, dimension(3,2)             :: group_size
      !> To check if group size is initialized
      logical                             :: group_init = .false.
      !> To concatenate position in order to create unique mpi message tag
      integer, dimension(3,2)             :: tag_size
      !> To concatenate rank in order to create unique mpi message tag
      integer                             :: tag_rank
      !> To check if parameter is already initialized
      logical                             :: mesh_init = .false.
    end type group_info


    ! ===== Public variables =====

    ! ----- Communicators -----
    !> Communicator associated with the cartesian topology
    integer, protected                  :: main_comm
    !> Communicator associated with the cartesian topology
    integer, protected                  :: cart_comm
    !> Communicators devoted to 1-dimensionnal subgrids (along Y and Z)
    integer, protected                  :: X_comm, Y_comm, Z_comm
    !> Table of the previous communicators (ie comm devoted to 1D subgrids)
    integer, dimension(3), protected    :: D_comm
    !> Rank of immediate neighbors
    integer,dimension(3,-4:4),protected :: neighbors
    !> Rank of immediate neighbors
    integer,dimension(1:3,-1:1),protected :: neighbors_cart_topo=0

    ! ----- Information about current MPI processus and MPI topology
    !> number of processes in each direction
    integer, dimension(3), protected    :: nb_proc_dim
    !> rank of current processus (in the cartesian communicator)
    integer, public                     :: cart_rank
    !> rank of current processus (in the in communicator associated to the different direction)
    integer, dimension(3), public       :: D_rank
    !> coordinate of the current processus
    integer, dimension(3), protected    :: coord
    !> YZ coordinate of the current processus
    integer, dimension(2), protected    :: coordYZ
    !> Periodic boundary conditions: logical array, equals true if periodic
    logical, dimension(3),protected     :: periods

    ! ------ Information about mesh subdivision and on the global grid -----
    !> information about local mesh - for scalar
    type(cartesian_mesh), protected     :: mesh_sc
    !> REcopy of mesh_sc%N_proc for python interface
    integer, dimension(3)               :: N_proc
    !> Computation are done by group of line. Here we define their size
    integer, dimension(3,2), protected  :: group_size
    !> To check if group size is initialized
    logical, private                    :: group_init = .false.
    !> To concatenate position in order to create unique mpi message tag
    integer, dimension(3,2), private    :: tag_size
    !> To concatenate rank in order to create unique mpi message tag
    integer, private                    :: tag_rank
    !> To check if mesh is already initialized
    logical, private                    :: mesh_init = .false.
    !> Default mesh resolution
    integer, parameter                  :: default_size = 80
    !> information about local mesh - for velocity
    type(cartesian_mesh), protected     :: mesh_V
    !> To check if mesh is already initialized
    logical, private                    :: mesh_velo_init = .false.


    ! ==== Public procedures ====
    ! Creation of the cartesian topology
    public      :: cart_create
    ! Initialise mesh information (first part)
    public      :: discretisation_create
    public      :: discretisation_default
    ! Compute tag for mpi message
    public      :: compute_tag
    private     :: compute_tag_gap
    private     :: compute_tag_NP
    ! Adjust some private variale
    public      :: set_group_size
    private     :: set_group_size_1
    private     :: set_group_size_1x2
    private     :: set_group_size_3
    private     :: set_group_size_init
    ! Create a cartesian_mesh variable related to data save in cart_topolgoy module.
    public      :: mesh_save_default

    ! ==== Public procedures ====
    ! Initialise mesh information (second part)
    private     :: discretisation_init

    interface compute_tag
        module procedure compute_tag_gap, compute_tag_NP
    end interface compute_tag

    interface set_group_size
    !>    Size of group of line is used to gather line together in the particle
    !! solver. As it is a crucial parameter, it must be possible for user to changed
    !! it without broke everything (ie if user value is bad it has to be ignored) and
    !! to be set up by default to a "right and intelligent" value. Use
    !! set_group_size for this purpose. An optional logical argument "init" is
    !! used to set up group_size to a default "acceptable" value if user value
    !! is not acceptable considering mesh size (ie will create bugs).
        module procedure set_group_size_1, set_group_size_1x2, set_group_size_3, &
            & set_group_size_init
    end interface set_group_size

contains

!> Creation of the cartesian mpi topology and (if needed) of all communicators
!! used for particles method.
!!    @param[in]    dims        = array specifying the number of processes in each dimension
!!    @param[in]    spec_comm   = main communicator
!!    @param[out]   ierr        = error code
!!    @param[out]   spec_comm   = mpi communicator used by the spectral part of the code (optional).
!!    @param[in]    topology    = to choose the dimension of the mpi topology (if 0 then none) (optional).
!! @details
!!        This subroutine initialzed the mpi topologic and returns the communicator
!!    that will be used for all the spectral part of the code (ie everything except
!!    the particles part). If needed, it also initialzed all the mpi context
!!    used by the particles solver.
subroutine cart_create(dims, ierr, parent_comm, spec_comm, topology)

    ! Input/Output
    integer, dimension(:), intent(in)   :: dims
    integer, intent(out)                :: ierr
    integer, intent(in)                 :: parent_comm
    integer, optional, intent(out)      :: spec_comm
    integer, optional, intent(in)       :: topology
    ! Other local variables
    logical                 :: reorganisation                   ! to choose to reordered or not the processus rank.
    logical, dimension(3)   :: remains_dim                      ! use to create 1D-subdivision : remains_dims(i) equal
                                                                ! true if the i-th dimension is kept in the subgrid.
    integer                 :: direction                        ! current direction : 1 = along X, 2 = along Y and 3 = alongZ
    integer                 :: topology_dim=3                   ! recopy of the optional input "topology".
    integer                 :: key                              ! to re-order processus in spec_comm
    integer, dimension(1)   :: nb_proc                          ! total number of processus
    logical, dimension(1)   :: period_1D = .false.              ! periodicity in case of 1D mpi topology.

    ! Duplicate parent_comm
    call mpi_comm_dup(parent_comm, main_comm, ierr)

    ! If there is some scalar to advec with particles method, then initialized
    ! the 2D mpi topology
    if (present(topology))  then
        select case (topology)
            case(0)
                topology_dim = 0
            case(1)
                topology_dim = 1
            case default
                topology_dim = 3
        end select
    end if

    select case (topology_dim)
    case (3)
        ! ===== Create a 2D mpi topology =====
        ! 2D topology is created and mpi context is initialized for both
        ! spectral and particles code

        ! --- Creation of the cartesian topology ---
        reorganisation = .true.
        periods = .true.
        if (size(dims)==2) then
            nb_proc_dim = (/ 1, dims(1), dims(2) /)
        else if (size(dims)==3) then
            nb_proc_dim = dims
            if (nb_proc_dim(1)/=1) then
                call mpi_comm_rank(main_comm, cart_rank, ierr)
                if (cart_rank==0) write(*,'(a)') ' XXXXXXXXXX Warning: subdision along X XXXXXXXXXX'
            end if
        else
            call mpi_comm_rank(main_comm, cart_rank, ierr)
            if (cart_rank==0) then
                write(*,'(a)') ' XXXXXXXXXX Error - wrong nb of processus XXXXXXXXXX'
                write(*,'(a,10(x,i0))') ' input argument dims =', dims
            end if
            stop
        end if

        call mpi_cart_create(main_comm, 3, nb_proc_dim, periods, reorganisation, &
                & cart_comm, ierr)

        ! --- Create 1D communicator ---
        ! Subdivision in 1D-subgrids and creation of communicator devoted to
        ! 1D-communication
        ! Communication along X-axis
        remains_dim = (/.true., .false., .false. /)
        call mpi_cart_sub(cart_comm, remains_dim, X_comm, ierr)
        D_comm(1) = X_comm
        ! Communication along Y-axis (in the 3D mesh, ie the x-axis on the mpi-topology)
        remains_dim = (/.false., .true., .false. /)
        call mpi_cart_sub(cart_comm, remains_dim, Y_comm, ierr)
        D_comm(2) = Y_comm
        ! Communication along Z-axis
        remains_dim = (/ .false., .false., .true. /)
        call mpi_cart_sub(cart_comm, remains_dim, Z_comm, ierr)
        D_comm(3) = Z_comm

        ! --- Initialise information about the current processus ---
        call mpi_comm_rank(cart_comm, cart_rank, ierr)
        do direction = 1, 3
            !neighbors on 1D topology
            call mpi_comm_rank(D_comm(direction), D_rank(direction), ierr)
            call mpi_cart_shift(D_comm(direction), 0, 1, neighbors(direction,-1), neighbors(direction,1), ierr)
            call mpi_cart_shift(D_comm(direction), 0, 2, neighbors(direction,-2), neighbors(direction,2), ierr)
            call mpi_cart_shift(D_comm(direction), 0, 3, neighbors(direction,-3), neighbors(direction,3), ierr)
            call mpi_cart_shift(D_comm(direction), 0, 3, neighbors(direction,-4), neighbors(direction,4), ierr)
            neighbors(direction,0) = D_rank(direction)
            !neighbors on 3D topology
            call mpi_cart_shift(cart_comm , direction-1, 1, neighbors_cart_topo(direction,-1), neighbors_cart_topo(direction,1), ierr)
            neighbors_cart_topo(direction,0) = cart_rank
        end do
        call mpi_cart_coords(cart_comm, cart_rank, 3, coord, ierr)
        coordYZ = (/ coord(2), coord(3) /)
        ! --- Spectral context ---
        ! Initialized the communicator used on which the spectral part
        ! will be based.
        if (present(spec_comm)) then
            !> Rank numerotation in spectral communicator grow along first
            !! direction first and then along the second, the opposite of mpi
            !! rank numerotation. That is why processus are reoder and 2
            !! communicator are created.
            !! Example with 4 processus
            !! coord    // mpi-cart rank    // spec rank
            !! (0,0,0)  // 0                // 0
            !! (0,1,0)  // 2                // 1
            !! (0,0,1)  // 1                // 2
            !! (0,1,1)  // 3                // 3
            ! Construct key to reoder
            key = coord(1) + (coord(2) + coord(3)*nb_proc_dim(2))*nb_proc_dim(1)
            ! As not split along X, it must be equivalent to "key = coord(2) + coord(3)*nb_proc_dim(2)"
            ! Construct spectral communicator
            call mpi_comm_split(cart_comm, 1, key, spec_comm, ierr)
        end if

    case (1)
            ! Construct 1D non-periodic mpi topology
            nb_proc = product(nb_proc_dim)
            call mpi_cart_create(main_comm, 1, nb_proc, period_1D, reorganisation, &
                & cart_comm, ierr)
            ! Use it as spectral communicator.
            spec_comm = cart_comm

    case default
        ! ===== Do not use mpi topology =====
        if (present(spec_comm)) then
            spec_comm = main_comm
        end if
        call mpi_comm_rank(main_comm,cart_rank,ierr)
    end select


    ! Print some minimal information about the topology
    if (cart_rank == 0) then
        write(*,'(a)') ''
        write(*,'(6x,a)') '========== Topology used ========='
        if (topology_dim == 0) then
            write(*,'(6x,a)') 'No mpi topology'
        else
            write(*,'(6x,i0,a)') topology_dim,'D mpi topology'
        end if
        write(*,'(6x,a,i0,x,i0,x,i0)') 'nb of proc along X, Y, Z = ', nb_proc_dim
        write(*,'(6x,a)') '=================================='
        write(*,'(a)') ''
    end if

end subroutine cart_create

!> Create the mesh structure associated to the topology
!!    @param[in]    Nx          = number of meshes along X
!!    @param[in]    Ny          = number of meshes along X
!!    @param[in]    Nz          = number of meshes along X
!!    @param[in]    Lx          = number of meshes along X
!!    @param[in]    Ly          = number of meshes along Y
!!    @param[in]    Lz          = number of meshes along Z
!!    @param[in]    verbosity   =  logical to unactivate verbosity (show message about group size change or not)
!! @details
!!    Initialise the mesh data associated to the mpi topology and used by the
!!    particle solver
!!    @author Jean-Baptiste Lagaert
subroutine discretisation_create(Nx, Ny, Nz, Lx, Ly, Lz, verbosity)

    ! Input/Output
    integer, intent(in)             :: Nx, Ny, Nz
    real(WP), intent(in)            :: Lx, Ly, Lz
    logical, intent(in), optional   :: verbosity    ! To unactivate verbosity

    ! Others
    logical                 :: show_message

    ! Init verbosity parameter
    show_message = .true.
    if(present(verbosity)) show_message = verbosity

    ! A cubic geometry : unitary lengh and 100 mesh points in each direction.
    mesh_sc%N(1) = Nx
    mesh_sc%N(2) = Ny
    mesh_sc%N(3) = Nz

    mesh_sc%length(1)= Lx
    mesh_sc%length(2)= Ly
    mesh_sc%length(3)= Lz

    mesh_sc%N_proc = mesh_sc%N / nb_proc_dim
    N_proc = mesh_sc%N_proc
    mesh_sc%relative_extend(:,1) = 1
    mesh_sc%relative_extend(:,2) = mesh_sc%N_proc

    ! Adjust group size :
    call set_group_size_init()
    ! Finish init
    mesh_init = .false.
    call discretisation_init(show_message)

end subroutine discretisation_create

!> Defaut mesh setup
!! @author Jean-Baptiste Lagaert
!!    @param[in]    verbosity   =  logical to unactivate verbosity (show message about group size change or not)
!! @details
!!    Initialise the mesh data associated to the mpi topology and used by the
!!    particle solver to a default 100x100x100 mesh grid.
subroutine discretisation_default(verbosity)

    logical, intent(in), optional   :: verbosity    ! To unactivate verbosity

    logical                 :: show_message

    ! Init verbosity parameter
    show_message = .true.
    if(present(verbosity)) show_message = verbosity

    ! A cubic geometry : unitary lengh and 100 mesh points in each direction.
    mesh_sc%N = default_size
    mesh_sc%length = 1.
    mesh_sc%N_proc = mesh_sc%N / nb_proc_dim
    N_proc = mesh_sc%N_proc
    mesh_sc%relative_extend(:,1) = 1
    mesh_sc%relative_extend(:,2) = mesh_sc%N_proc

    group_init = .false.
    call set_group_size_init()
    mesh_init = .false.
    call discretisation_init(show_message)

end subroutine discretisation_default

!> To initialize some hidden mesh parameters
!! @author Jean-Baptiste Lagaert
!!    @param[in]    verbosity   = optional, logical used to unactivate verbosity
!! @details
!!        In order to deal well with the mpi topology, the data structure and the
!!    mesh cut, some other parameters have to be initialised. Some are parameters
!!    that could not be choose by the user (eg the space step which depend of the
!!    domain size and the number of mesh) and some other are "hidden" parameter used
!!    to avoid communication error or to allowed some optimization. For example, it
!!    include variable used to create unique tag for the different mpi communication,
!!    to gather line in group and to index these group.
subroutine discretisation_init(verbosity)

    logical, intent(in), optional   :: verbosity    ! To unactivate verbosity

    integer                 :: direction    ! direction (along X = 1, along Y = 2, along Z = 3)
    integer                 :: group_dir    ! direction "bis"
    integer, dimension(3,2) :: N_group      ! number of group on one processus along one direction
    logical                 :: show_message

    mesh_sc%dx = mesh_sc%length/(mesh_sc%N)
    show_message = .true.
    if(present(verbosity)) show_message = verbosity

    ! Compute number of group
    ! Group of line along X
    N_group(1,1) = mesh_sc%N_proc(2)/group_size(1,1)
    N_group(1,2) = mesh_sc%N_proc(3)/group_size(1,2)
    ! Group of line along X
    N_group(2,1) = mesh_sc%N_proc(1)/group_size(2,1)
    N_group(2,2) = mesh_sc%N_proc(3)/group_size(2,2)
    ! Group of line along X
    N_group(3,1) = mesh_sc%N_proc(1)/group_size(3,1)
    N_group(3,2) = mesh_sc%N_proc(2)/group_size(3,2)

    ! tag_size = smallest power of ten to ensure tag_size > max ind_group
    do direction = 1,3
        tag_size(direction,:) = 1
        do group_dir = 1,2
            do while (N_group(direction, group_dir)/(10**tag_size(direction, group_dir))>1)
                tag_size(direction, group_dir) = tag_size(direction, group_dir)+1
            end do
        end do
    end do

    tag_rank = 1
    do while(3*max(nb_proc_dim(1),nb_proc_dim(2),nb_proc_dim(3))/(10**tag_rank)>=1)
        tag_rank = tag_rank+1
    end do
    if (tag_rank == 1) tag_rank = 2

    ! Default velocity mesh = same mesh than scalar
    mesh_V = mesh_sc

    ! Print some information about mesh used
    if((cart_rank==0).and.(show_message)) then
        write(*,'(a)') ''
        if(mesh_init) then
            write(*,'(6x,a,a24,a)') 'XXXXXX','group sized changed ','XXXXXX'
        else
            write(*,'(6x,a,a30,a)') '-- ','mesh size',' --'
            write(*,'(6x,a,3(x,i0))') 'global size =',mesh_sc%N
            write(*,'(6x,a,3(x,i0))') 'local size =',mesh_sc%N_proc
        end if
        write(*,'(6x,a,2(x,i0))') 'group size along X =',group_size(1,:)
        write(*,'(6x,a,2(x,i0))') 'group size along Y =',group_size(2,:)
        write(*,'(6x,a,2(x,i0))') 'group size along Z =',group_size(3,:)
        write(*,'(6x,a)') '-- initialisation: tag generation --'
        do direction = 1,3
            write(*,'(6x,a,i0,a,i0,x,i0)') 'tag_size(',direction,',:) = ', tag_size(direction,:)
        end do
        write(*,'(6x,a,i0)') 'tag_rank = ', tag_rank
        write(*,'(6x,a)') '------------------------------------'
        write(*,'(a)') ''
    end if

    mesh_init = .true.

end subroutine discretisation_init

!> To change velocity resolution
!!    @param[in] Nx   = number of points along X
!!    @param[in] Ny   = number of points along Y
!!    @param[in] Nz   = number of points along Z
subroutine discretisation_set_mesh_Velo(Nx, Ny, Nz)

    integer, intent(in) :: Nx, Ny, Nz

    mesh_V%N(1) = Nx
    mesh_V%N(2) = Ny
    mesh_V%N(3) = Nz

    mesh_V%N_proc = mesh_V%N / nb_proc_dim
    mesh_V%relative_extend(:,2) = mesh_V%N_proc

    mesh_V%dx = mesh_V%length/(mesh_V%N)

end subroutine discretisation_set_mesh_Velo

!> Compute unique tag for mpi message by concatenation of position (ie line coordinate), proc_gap and unique Id
!!    @param[in]    ind_group   = indice of current group of line
!!    @param[in]    tag_param   = couple of int unique for each message (used to create the tag)
!!    @param[in]    direction   = current direction
!!    @param[in]    proc_gap    = number of processus between the sender and the receiver
!!    @return       tag         = unique tag: at each message send during an iteration have a different tag
!!@details
!!     Use this procedure to compute tag in order to communicate with a distant processus or/and when
!!    you will send more then two message. It produce longer tag compute_tag_NP because rather tyo use 0/1 it
!!    put the gap between the sender and the receiver (ie the number of processus between them) in the tag.
!!    Using these two procedure allow to obtain more unique tag for communication.
function compute_tag_gap(ind_group, tag_param, direction,proc_gap) result(tag)

    ! Returned variable
    integer                             :: tag
    ! Input/Ouput
    integer, dimension(2), intent(in)   :: ind_group
    integer, dimension(2), intent(in)   :: tag_param
    integer, intent(in)                 :: direction
    integer, intent(in)                 :: proc_gap
    ! Other local variables
    integer                              :: abs_proc_gap ! absolute value of proc_gap

    abs_proc_gap = max(abs(proc_gap),1)
    tag = (tag_param(1)*10+direction)*(10**(tag_rank+1))
    if (proc_gap>=0) then
        tag = tag + proc_gap*10
    else
        tag = tag - proc_gap*10 +1
    end if
    tag = (tag*(10**tag_size(direction,1)))+(ind_group(1)-1)
    tag = ((tag*(10**tag_size(direction,2)))+(ind_group(2)-1))
    tag = (tag*10)+tag_param(2)

    ! As tag can not be to big (it must be a legal integer and smaller than
    ! maximum mpi tag)
    if ((tag<0).or.(tag>MPI_TAG_UB))  then
        !print*, 'tag too big - regenerated'
        tag = (tag_param(1))*(10**(tag_rank+1))
        if (proc_gap>=0) then
            tag = tag + proc_gap*10
        else
            tag = tag - proc_gap*10 +1
        end if
        tag = tag*(10**tag_size(direction,1))+(ind_group(1)-1)
        tag = ((tag*(10**tag_size(direction,2)))+(ind_group(2)-1))
        tag = (tag*10)+tag_param(2)
        if ((tag<0).or.(tag>MPI_TAG_UB))  then
            !print*, 'tag very too big - regenerated'
            tag = (tag_param(1))*(10**(tag_rank+1))
            if (proc_gap>=0) then
                tag = tag + proc_gap*10
            else
                tag = tag - proc_gap*10 +1
            end if
            tag = (tag*10)+tag_param(2)
            if ((tag<0).or.(tag>MPI_TAG_UB))  then
                tag = tag_param(1)*10 + tag_param(2)
                if (proc_gap<0) tag = tag +100
                !print*, 'rank = ', cart_rank, ' coord = ', coord
                !print*, 'ind_group = ', ind_group, ' ; tag_param = ', tag_param
                !print*, 'direction = ', direction, ' gap = ', proc_gap ,' and tag = ', tag
            end if
        end if
    end if
! XXX Fin aide au debug XXX

end function compute_tag_gap


!> Compute unique tag for mpi message by concatenation of position(ie line coordinate), +1 or -1 and unique Id
!!    @param[in]    ind_group   = indice of current group of line
!!    @param[in]    tag_param   = couple of int unique for each message (used to create the tag)
!!    @param[in]    direction   = current direction
!!    @return       tag_table   = unique couple tag: use tag_table(1) for mesage to previous proc. (or first
!!                                message ) and tag_table(2) for the other message.
!!@details
!!     Use this procedure to compute tag for communication with your neighbor or when only two message are send:
!!    it produce smaller tag then compute_tag_gap because the gap between sender and receiver are replaced by 1,
!!    for communicate with previous processus (or first of the two message), or 0, for communication with next
!!    processus (or the second message). It allow to reuse some unique Id.
function compute_tag_NP(ind_group, tag_param, direction) result (tag_table)

  ! Returned variable
    integer, dimension(2)               :: tag_table
    ! Input/Ouput
    integer, dimension(2), intent(in)   :: ind_group
    integer, dimension(2), intent(in)   :: tag_param
    integer, intent(in)                 :: direction

    tag_table(2) = (tag_param(1)*10+direction)*10
    tag_table(1) = tag_table(2)

    tag_table(2) = tag_table(2) +1

    tag_table(2) = (tag_table(2)*(10**tag_size(direction,1)))+(ind_group(1)-1)
    tag_table(1) = (tag_table(1)*(10**tag_size(direction,1)))+(ind_group(1)-1)

    tag_table(2) = ((tag_table(2)*(10**tag_size(direction,2)))+(ind_group(2)-1))
    tag_table(1) = ((tag_table(1)*(10**tag_size(direction,2)))+(ind_group(2)-1))

    tag_table(2) = (tag_table(2)*10)+tag_param(2)
    tag_table(1) = (tag_table(1)*10)+tag_param(2)

    ! Check if tag limitations are respected.
    if ((minval(tag_table)<0).or.(maxval(tag_table)>MPI_TAG_UB))  then
        tag_table = tag_param(1)*100 + tag_param(2)*10
        tag_table = tag_table + (/1,2/)
        !print*, 'rank = ', cart_rank, ' coord = ', coord
        !print*, 'ind_group = ', ind_group, ' ; tag_param = ', tag_param
        !print*, 'direction = ', direction, ' and tag = ', tag_table
    end if


end function compute_tag_NP


!> Adjust the private variable "group_size": line are gathering on group of same
!! size undependant from the direction
!!    @param[in]    s           =  integer such as group will gather sxs lines
!!    @param[in]    init        =  logical to said if it is a default init of group_size
!!    @param[in]    verbosity   =  logical to unactivate verbosity (show message about group size change or not)
!! @details
!!    Create group of line s x s along the three direction.
subroutine set_group_size_1(s, init, verbosity)

    integer, intent(in)             :: s
    logical, intent(in), optional   :: init
    logical, intent(in), optional   :: verbosity

    if (.not.mesh_init) then
        group_size = s
        ! And now group size is initialized !
        group_init = .true.
    else
        if (all(mod(mesh_sc%N_proc,s)==0)) group_size = s
        if (present(verbosity)) then
            call discretisation_init(verbosity=verbosity)
        else
            call discretisation_init()
        end if
    end if

    if (present(init)) call set_group_size(init)

end subroutine set_group_size_1


!> Adjust the private variable "group_size": line are gathering on group of same
!! size undependant from the direction
!!    @param[in]    s1      =  integer such as group will gather s1 line along first remaining direction
!!    @param[in]    s2      =  integer such as group will gather s1 line along second remaining direction
!!    @param[in]    init    =  logical to said if it is a default init of group_size
!!    @param[in]    verbo   =  logical to unactivate verbosity (show message about group size change or not)
!! @details
!!    Created group will gather s1 x s2 lines
subroutine set_group_size_1x2(s1, s2, init, verbo)

    integer, intent(in)             :: s1, s2
    logical, intent(in), optional   :: init
    logical, intent(in), optional   :: verbo

    if (.not. mesh_init) then
        group_size(:,1) = s1
        group_size(:,2) = s2
        ! And now group size is initialized !
        group_init = .true.
    else
        if (all(mod(mesh_sc%N_proc,s1)==0)) group_size(:,1) = s1
        if (all(mod(mesh_sc%N_proc,s2)==0)) group_size(:,2) = s2
        if (present(verbo)) then
            call discretisation_init(verbosity=verbo)
        else
            call discretisation_init()
        end if
    end if

    if (present(init)) call set_group_size(init)

end subroutine set_group_size_1x2


!> Adjust the private variable "group_size": line are gathering on group of a
!! size depending of the current direction.
!!    @param[in]    sX =  integer such as group of lines along X will gather sX x sX lines
!!    @param[in]    sY =  integer such as group of lines along Y will gather sY x sY lines
!!    @param[in]    sZ =  integer such as group of lines along Z will gather sZ x sX lines
!!    @param[in]    init    =  logical to said if it is a default init of group_size
!!    @param[in]    verbo   =  logical to unactivate verbosity (show message about group size change or not)
subroutine set_group_size_3(sX, sY, sZ, init, verbo)

    integer, intent(in)             :: sX, sY, sZ
    logical, intent(in), optional   :: init
    logical, intent(in), optional   :: verbo

    if (.not.mesh_init) then
        group_size(1,:) = (/sY, sZ/)
        group_size(2,:) = (/sX, sZ/)
        group_size(3,:) = (/sX, sY/)
        ! And now group size is initialized !
        group_init = .true.
    else
        if (all(mod(mesh_sc%N_proc(2:3),sX)==0)) group_size(1,:) = sX
        if ((mod(mesh_sc%N_proc(1),sY)==0).and.(mod(mesh_sc%N_proc(3),sY)==0)) group_size(2,:) = sY
        if ((mod(mesh_sc%N_proc(1),sZ)==0).and.(mod(mesh_sc%N_proc(2),sZ)==0)) group_size(3,:) = sZ
        if (present(verbo)) then
            call discretisation_init(verbosity=verbo)
        else
            call discretisation_init()
        end if
    end if

    if (present(init)) call set_group_size(init)


end subroutine set_group_size_3

!> Adjust the private variable "group_size": line are gathering on group of same
!! size undependant from the direction
!!    @param[in]    init    =  logical to said if it is a default init of group_size
!! @details
!!    Create group of acceptable default size (or re-init group size if optional
!! argument "init" is present and set to true).
subroutine set_group_size_init(init)

    logical, intent(in), optional   :: init

    ! To check if group size is well defined
    integer, dimension(3,2)         :: domain_size

    if (present(init)) group_init = init

    if (.not.group_init) then
        ! Setup the size of line group to a default value
        if (all(mod(mesh_sc%N_proc,8)==0)) then
            group_size = 8
        else if (all(mod(mesh_sc%N_proc,5)==0)) then
            group_size = 5
        else if (all(mod(mesh_sc%N_proc,4)==0)) then
            group_size = 4
        else if (all(mod(mesh_sc%N_proc,2)==0)) then
            group_size = 2
        else
            group_size = 1
        end if
        ! And now group size is initialized !
        group_init = .true.
    else
        domain_size(1,:) = (/mesh_sc%N_proc(2), mesh_sc%N_proc(3)/)
        domain_size(2,:) = (/mesh_sc%N_proc(1), mesh_sc%N_proc(3)/)
        domain_size(3,:) = (/mesh_sc%N_proc(1), mesh_sc%N_proc(2)/)

        where (mod(domain_size,group_size)/=0)
            where(mod(domain_size,8)==0)
                group_size=8
            elsewhere(mod(domain_size,5)==0)
                group_size=5
            elsewhere(mod(domain_size,4)==0)
                group_size=4
            elsewhere(mod(domain_size,2)==0)
                group_size=2
            elsewhere
                group_size=1
            end where
        end where
    end if

end subroutine set_group_size_init


!> Save data about the cartesian mesh create in cart_topology module
!>    @param[out]   mesh    = varialbe of type cartesian_mesh where the data about mesh are save
subroutine mesh_save_default(mesh)

    ! Input/Output
    type(cartesian_mesh), intent(out)       :: mesh
    ! Other local variables
    !integer                                 :: direction    ! integer matching to a direction (X, Y or Z)

    mesh = mesh_sc

end subroutine mesh_save_default



end module cart_topology
!> @}
