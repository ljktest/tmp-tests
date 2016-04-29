!> @addtogroup output
!! @{

!------------------------------------------------------------------------------
!
! MODULE: parallel_out
!
!
! DESCRIPTION: 
!> This module provide all procedure needed to perform parallel and distribued
!! output at vtk format. All data ouptut are binary output rather than ascii
!! output.
!
!! @details
!!        This module developp tools to write distribued output. This means that
!!    output of one field will be done in one file per (mpi) processus. These allows
!!    to write field computed with an high resolution without be limitated by the
!!    size of the file output and to avoid to big loading time during visualisation
!!    or file loading in order to initialize computation at a given setup.
!!
!!         This first version provide only output tools. Some input procedures could
!!    be add in future works. The general context (number of field to save, physical 
!!    dimension, number of processus, ...) is initiliazed by calling "vtkxml_init_all"
!!    the context specific to each field (mesh resolution, number of point, name of
!!    the ouput, information about time sequence, ...) is initialized or save by
!!    calling "vtkxml_init_field". After that, call "parallel_write" in order to
!!    create a new output of the field.
!
!> @author
!! Jean-Baptiste Lagaert, LEGI
!
!------------------------------------------------------------------------------

module vtkxml_bin

    use precision_tools
    use cart_mesh_tools

    implicit none


    interface parallel_write
        !module procedure parallel_vect3D, parallel_scalar
        module procedure parallel_scalar
    end interface parallel_write


    ! ===== Parameter about name size =====
    !> Size of character string for the name of each field (name of output will be longer)
    integer, private, parameter :: short_name = 12
    !> Number of character use to write the indice of an output
    integer, private, parameter :: size_ite = 3
    !> Number of character use to write the rank of the processus
    integer, private            :: size_rank = 3
    !> Size of character string corresponding to the final output name
    integer, private            :: long_name = short_name + size_ite + 8

    ! ===== Type =====
    ! > Information about data to save
    type io_data
        !> field name
        character (len=short_name)              :: f_name
        !> indice of the current output - in order to construct time sequence.
        integer                                 :: iteration
        !> mesh information
        type(cartesian_mesh)                    :: mesh
        !> piece extend (for distribued vtk xml format)
        integer, dimension(:,:,:), allocatable  :: piece_extend
    end type io_data

    ! ===== Public procedures =====
    !> Generique procedure to write field in distribued vtk files
    public                  :: parallel_write
    ! Initialisation of the mesh context
    public                  :: vtkxml_init_all
    public                  :: vtkxml_init_field
    ! Destruct the io context and free memory
    public                  :: vtkxml_finish


    ! ===== Private procedures =====
    ! Specifique procedure to write field in distribued vtk files
    !private                 :: parallel_vect3D
    private                 :: parallel_scalar
    ! To search the right tag if the input one does not match to the field name.
    private                 :: search_io_tag


    ! ===== Private variable =====
    !> number of processes in each direction
    integer, dimension(3), private                      :: nb_proc
    !> total number of processes 
    integer, private                                    :: total_nb_proc
    !> space lengh of th domain
    real(WP), dimension(3), private                     :: domain_length
    !> coordinate of the current processus
    integer, dimension(3), private                      :: coord3D
    !> number of different field to save
    integer, private                                    :: field_number
    !> number of field for which io_data are already initialized
    integer, private                                    :: field_initialized
    !> io_data for each field to save
    type(io_data), dimension(:), allocatable, private   :: field_data
    !> rank of current processus (in the 3D cartesian communicator)
    integer, private                                    :: rank3D
    !> write format for output name of *.pvti (file without data listing the distribued output)
    character (len=100), private                        :: format_no_rank
    !> write format for output name of *.vti (distribued file containing the data)
    character (len=100), private                        :: format_rank
    !> write format for datas (ie array output)
    character (len=100), private                        :: format_data
    !> directory where data are saved
    character (len=50), private                         :: dir = './save/'
    !> smallest value which could be write
    real, parameter, private                         :: minValue = 1e-9
    !> Format of output (ascii or binary)
    logical, private                                    :: binary = .true.

    contains

!> Initialize the general io context
!!    @param[in]    nb_field    = number of different field to save
!!    @param[in]    nb_proc_3D  = number of processus along the different direction
!!    @param[in]    length_3D   = size of the (physical or eventually spectral) domain
!!    @param[in]    rank_3D     = rank of the current processus in the 3D mpi-topology
!!    @param[in]    coord_3D    = coordinates of the current processus in the 3D mpi-topology
!!    @param[in]    dir_name    = (optional) name of the directory where output have to be done
subroutine vtkxml_init_all(nb_field, nb_proc_3D, length_3D, rank_3D, coord_3D, dir_name)

    implicit none

    integer, intent(in)                     :: nb_field, rank_3D
    integer, dimension(3), intent(in)       :: nb_proc_3D, coord_3D
    real(WP), dimension(3), intent(in)      :: length_3D
    character (len=*), intent(in), optional :: dir_name

    
    ! Update save directory
    if (present(dir_name)) dir = trim(dir_name)
    
    ! copy data
    field_number    = nb_field
    nb_proc         = nb_proc_3D
    domain_length   = length_3D
    rank3D          = rank_3D
    coord3D         = coord_3D

    ! Allocate field_data
    allocate(field_data(field_number))
    field_initialized = 0
    
    ! Compute "size_rank"
    total_nb_proc = nb_proc(1)*nb_proc(2)*nb_proc(3)
    size_rank = 1
    do while (total_nb_proc/(10**size_rank)>1)
        size_rank = size_rank + 1
    end do
    long_name = long_name + size_rank + len(trim(dir))

    ! Init format for output file name
    write(format_no_rank,'(a,i0,a1,i0,a)') '(a,a1,i',size_ite,'.',size_ite,',a)'
    write(format_rank,'(a,i0,a1,i0,a,i0,a1,i0,a)') '(a,a2,i',size_rank,'.',size_rank, &
            & ',a1,i',size_ite,'.',size_ite,',a)'

end subroutine vtkxml_init_all


!> Initialize the context associated to a given field
!!    @param[in]    f_name  = name of the field (as it will appear in the output)
!!    @param[out]   tag         = tag used to identifiate the field from the others one when using "parallel_write"
!!    @param[in]    mesh        = mesh data associated to the current field (optional, if not present use default one)
subroutine vtkxml_init_field(f_name, tag, mesh)

    
    use cart_topology

    implicit none

    character (len=*), intent(in)               :: f_name
    type(cartesian_mesh), intent(in), optional  :: mesh
    integer, intent(out)                        :: tag

    integer                                     :: rank         ! mpi processus rank
    integer, dimension(3)                       :: coord3D      ! coordinate of the processus in the 3D mpi topology
    integer                                     :: ierr         ! mpi error code
    integer                                     :: direction

    if (field_initialized<field_number) then 
        field_initialized = field_initialized + 1
    else
        stop 'more field to save than anounced in the construction of the io context (vtk parallel output)'
    end if

    tag = field_initialized

    ! Save io_data for the field
    field_data(tag)%f_name = f_name
    field_data(tag)%iteration  = 0
    if(present(mesh)) then
        field_data(tag)%mesh = mesh
    else
        call mesh_save_default(field_data(tag)%mesh)
    end if

    ! Compute piece extend
    allocate(field_data(tag)%piece_extend(0:total_nb_proc-1,3,2))
    do rank = 0, total_nb_proc-1
        call mpi_cart_coords(cart_comm, rank, 3, coord3D, ierr)
        do direction=1, 3
            field_data(tag)%piece_extend(rank,direction,1)=coord3D(direction)*field_data(tag)%mesh%N_proc(direction)+1
            if (coord(direction)<nb_proc(direction)-1) then
                field_data(tag)%piece_extend(rank,direction,2)=(coord3D(direction)+1)*field_data(tag)%mesh%N_proc(direction)+1
            else
                field_data(tag)%piece_extend(rank,direction,2)=(coord3D(direction)+1)*field_data(tag)%mesh%N_proc(direction)
            end if
        end do
    end do

end subroutine vtkxml_init_field


!> Destruct the io context and free memory.
subroutine vtkxml_finish()

    integer     :: tag      ! field identifiant

    ! Free memory
    do tag = 1, field_initialized
        deallocate(field_data(tag)%piece_extend)
    end do
    deallocate(field_data)

end subroutine vtkxml_finish


! ===========================================================
! ==================== Private procedures ===================
! ===========================================================

!> Write an output for an "one-component" field (eg scalar)
!!    @param[in,out]    tag     = tag associated to the field to save (ie matching indice in the table "field_data")
!!    @param[in]        values  = field (ie real values) to write
!!    @param[in]        f_name  = name of the field (optional, redondant with tag, can be used to check it)
subroutine parallel_scalar(tag, values, f_name)

    integer, intent(inout)                          :: tag
    character (len=*), intent(in), optional         :: f_name
    double precision, dimension(:,:,:), intent(in)  :: values
    
    character (len=long_name+size_rank)         :: file_name    ! output name
    integer                                     :: k, j, i      ! some array indices
    character (len=short_name)                  :: f_name_bis
    character (len=2000)                 :: buffer_char  ! to write in ascii in a file open in binary (for vtk header)
    
    ! Check tag for output
    if (present(f_name)) then
        f_name_bis = trim(f_name)
        if (field_data(tag)%f_name /= f_name_bis) tag = search_io_tag(f_name_bis)
    end if

    ! XXX TODO OR NOT ? Check mesh data ? XXX

    ! ===== Write data ===== 

    ! Write the parallel file wich contain information about distribution (but no datas)
    if (rank3D==0) call parallel_file_description(tag)

    ! Write local file
    ! Open file
    file_name = trim(dir)//trim(output_name(tag, rank3D))
    open(unit = 44, file=file_name, form='unformatted', access='stream')
!    if (binary) then
        !open(unit = 44, file=file_name, status='new', access='stream')
!        open(unit = 44, file=file_name, status='new', form='formatted')
!    else
!        open(unit = 44, file=file_name, status='new', form='formatted')
!    end if
    ! XXX TODO : on ouvre en binaire, puis quand on écrit des char, il fait bien
    ! du ascii !!! 
    ! Write header
    write(buffer_char, '(a21)') '<?xml version="1.0"?>'
    write(44) trim(buffer_char)
    write(44) char(10)
    write(buffer_char, '(a)') '<VTKFile type="ImageData" version="0.1" byte_order="LittleEndian">'
    write(44) trim(buffer_char)
    write(44) char(10)
    write(buffer_char,'(2x,a,i0,x,i0,x,i0,x,i0,x,i0,x,i0,a)') '<ImageData WholeExtent="', &
        & field_data(tag)%piece_extend(rank3D,1,1), &
        & field_data(tag)%piece_extend(rank3D,1,2), &
        & field_data(tag)%piece_extend(rank3D,2,1), &
        & field_data(tag)%piece_extend(rank3D,2,2), &
        & field_data(tag)%piece_extend(rank3D,3,1), &
        & field_data(tag)%piece_extend(rank3D,3,2),'"'
    write(44) trim(buffer_char)
    write(44) char(10)
    write(buffer_char, '(4x,a)') 'Origin="0 0 0"'
    write(44) trim(buffer_char)
    write(44) char(10)
    write(buffer_char, '(4x,a,f6.4,x,f6.4,x,f6.4,a)') 'Spacing="', field_data(tag)%mesh%dx(1), &
        & field_data(tag)%mesh%dx(2), field_data(tag)%mesh%dx(3), '">'
    write(44) trim(buffer_char)
    write(44) char(10)
    write(buffer_char, '(4x,a,i0,x,i0,x,i0,x,i0,x,i0,x,i0,a)') '<Piece Extent="', &
        & field_data(tag)%mesh%absolute_extend(1,1), &
        & field_data(tag)%mesh%absolute_extend(1,2), &
        & field_data(tag)%mesh%absolute_extend(2,1), &
        & field_data(tag)%mesh%absolute_extend(2,2), &
        & field_data(tag)%mesh%absolute_extend(3,1), &
        & field_data(tag)%mesh%absolute_extend(3,2),'">'
    write(44) trim(buffer_char)
    write(44) char(10)

    ! Write data
    ! Information about field
    write(buffer_char,'(6x,a,a,a)') '<PointData Scalars="', trim(field_data(tag)%f_name),'">'
    write(44) trim(buffer_char)
    write(44) char(10)
!    if (binary) then
        write(buffer_char,'(8x,a,a,a)') '<DataArray type="Float64" Name="', &
                & trim(field_data(tag)%f_name),'" NumberOfComponents="1" format="binary">'
        write(44) trim(buffer_char)
        write(44) char(10)

        do k = 1, field_data(tag)%mesh%N_proc(3)
            do j = 1, field_data(tag)%mesh%N_proc(2)
                do i = 1, field_data(tag)%mesh%N_proc(1)
                    write(44) values
                end do
            end do
        end do
        write(44) char(10)
!    else
!        write(buffer_char,'(8x,a,a,a)') '<DataArray type="Float64" Name="', &
!                & trim(field_data(tag)%f_name),'" NumberOfComponents="1" format="ascii">'
!        write(44) trim(buffer_char)
!        write(44) char(10)
!        ! Data in ascii format
!        write(format_data,'(a,i0,a)') '(',field_data(tag)%mesh%N_proc(1), '(f15.5,x))'
!
!        do k = 1, field_data(tag)%mesh%N_proc(3)
!            do j = 1, field_data(tag)%mesh%N_proc(2)
!                write(buffer_char, format_data) values(:,j,k)
!                write(*) buffer_char
!                write(44) char(10)
!            end do
!        end do
!    end if

    ! Close environment
    write(buffer_char,'(8x,a12)') '</DataArray>'
    write(44) trim(buffer_char)
    write(44) char(10)
    write(buffer_char,'(6x,a12)') '</PointData>'
    write(44) trim(buffer_char)
    write(44) char(10)

    ! Write footer
    write(buffer_char, '(4x,a8)') '</Piece>'
    write(44) trim(buffer_char)
    write(44) char(10)
    write(buffer_char, '(2x,a12)') '</ImageData>'
    write(44) trim(buffer_char)
    write(44) char(10)
    write(buffer_char, '(a10)') '</VTKFile>'
    write(44) trim(buffer_char)
    write(44) char(10)


    close(44)
    field_data(tag)%iteration = field_data(tag)%iteration + 1

end subroutine parallel_scalar


!> Write the parallel file wich contain information about distribution (but no datas) for parallel vtk output.
!!    @param[in,out]    tag     = tag associated to the field to save (ie matching indice in the table "field_data")
subroutine parallel_file_description(tag)

    integer, intent(in)         :: tag

    character (len=long_name)   :: file_name
    integer                     :: rank

    ! ===== Write data ===== 

    ! Open file
    file_name = trim(dir)//trim(output_name(tag))
    open(unit = 45, file=file_name, status='new', form='formatted')

    ! Write header
    write(45, '(a21)') '<?xml version="1.0"?>'
    write(45, '(a)') '<VTKFile type="PImageData" version="0.1" byte_order="LittleEndian">'
    write(45,'(2x,a,i1,x,i0,x,i1,x,i0,x,i1,x,i0,a)') '<PImageData WholeExtent="', &
            & 1, field_data(tag)%mesh%N(1), 1, field_data(tag)%mesh%N(2), 1, field_data(tag)%mesh%N(3), '"'
    write(45, '(4x,a)') 'GhostLevel="0"'
    write(45, '(4x,a)') 'Origin="0 0 0"'
    write(45, '(4x,a,f7.5,x,f7.5,x,f7.5,a)') 'Spacing="', field_data(tag)%mesh%dx(1), &
        & field_data(tag)%mesh%dx(2), field_data(tag)%mesh%dx(3), '">'
    ! Write information about data field
    write(45,'(6x,a,a,a)') '<PPointData Scalars="', trim(field_data(tag)%f_name),'">'
    write(45,'(8x,a,a,a)') '<PDataArray type="Float64" Name="', trim(field_data(tag)%f_name),'" NumberOfComponents="1" >'
    write(45,'(8x,a13)') '</PDataArray>'
    write(45,'(6x,a13)') '</PPointData>'

    ! Piece description
    do rank = 0, total_nb_proc-1
        write(45, '(4x,a,i0,x,i0,x,i0,x,i0,x,i0,x,i0,a)') '<Piece Extent="', &
            & field_data(tag)%piece_extend(rank,1,1), &
            & field_data(tag)%piece_extend(rank,1,2), &
            & field_data(tag)%piece_extend(rank,2,1), &
            & field_data(tag)%piece_extend(rank,2,2), &
            & field_data(tag)%piece_extend(rank,3,1), &
            & field_data(tag)%piece_extend(rank,3,2),'"'
        write(45, '(6x,a8,a,a)') 'Source="',trim(output_name(tag,rank)),'"/>'
    end do

    ! Write footer
    write(45, '(2x,a13)') '</PImageData>'
    write(45, '(a10)') '</VTKFile>'

    close(45)

end subroutine parallel_file_description


!> Search the tag associated to a given name
!!    @param[in]    f_name  = name of the field (as it will appear in the output)
!!    @return       tag     = tag associated to the field to save (ie matching indice in the table "field_data")
!> Search the tag associated to a given name
!!    @param[in]    f_name  = name of the field (as it will appear in the output)
!!    @return       tag     = tag associated to the field to save (ie matching indice in the table "field_data")
function search_io_tag(f_name) result(tag)

    character (len=short_name), intent(in)  ::  f_name
    integer                                 ::  tag

    tag = 1
    do while (field_data(tag)%f_name /= f_name)
        tag = tag + 1
        if (tag > field_number) then
            print *, 'wrong output name = ', f_name
            stop ' it is not usefull to continue computation without saving result'
        end if
    end do

end function search_io_tag


!> Create output name by concatenation of field name, rank (optional) and indice of the current output.
!!    @param[in]    tag     = tag associated to the field to save (ie matching indice in the table "field_data")
!!    @param[in]    rank    = rank of the current processus
!!    @return       io_name = name of the output
function output_name(tag, rank) result(io_name)
    
    integer, intent(in)                 :: tag
    integer, intent(in), optional       :: rank
    character (len=long_name)           :: io_name

    write(io_name,format_no_rank) trim(field_data(tag)%f_name), &
        & '-',field_data(tag)%iteration,'.pvti'
    if (present(rank)) then
        write(io_name,format_rank) trim(field_data(tag)%f_name), &
            &'_p',rank,'-',field_data(tag)%iteration,'.vti'
    end if

end function output_name

end module vtkxml_bin
!> @}
