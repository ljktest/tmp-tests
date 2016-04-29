!> Interface to scales-advection solver
module scales2py

use cart_topology, only : cart_create,set_group_size,discretisation_create,N_proc,coord,cart_rank,discretisation_set_mesh_Velo
use advec, only : advec_init,advec_step,advec_step_Inter_basic,advec_step_Inter_Two
use advec_vect, only : advec_step_Vect,advec_step_Inter_basic_Vect
use interpolation_velo, only : interpol_init
use mpi
use hysopparam


implicit none

contains

  !> Initialisation of advection solver (from scales) context : create topology and memory buffers
  !! @param[in] ncells number of cells in the global discrete domain
  !! @param[in] lengths width of each side of the domain
  !! @param[in] topodims number of mpi-processus in each dir
  !! @param[in] mpi communicator from python
  !! @param[out] datashape local dimension of the input/output field
  !! @param[out] offset absolute index of the first component of the local field
  subroutine init_advection_solver(ncells,lengths,topodims,main_comm,datashape,offset,dim,order,dim_split)
    integer, intent(in) :: dim
    integer, dimension(dim),intent(in) :: ncells
    real(pk),dimension(dim), intent(in) :: lengths
    integer, dimension(dim), intent(in) :: topodims
    integer, intent(in)                 :: main_comm
    integer(ik), dimension(dim), intent(out) :: datashape
    integer(ik), dimension(dim), intent(out) :: offset
    character(len=*), optional, intent(in)  ::  order, dim_split
    !! real(pk), optional, intent(out) ::  stab_coeff
    real(pk) ::  stab_coeff
    !f2py integer optional , depend(ncells) :: dim=len(ncells)
    !f2py intent(hide) dim
    !f2py character(*) optional, intent(in) :: order = 'p_O2'
    !f2py, depends(dim), intent(in) :: topodims
    integer :: error  !,groupsize !rank, nbprocs

    if(dim /= 3) then
       stop 'Scales advection solver initialisation failed : not yet implemented for 2d problem.'
    end if

    ! get current process rank
    !call MPI_COMM_RANK(MPI_COMM_WORLD,rank,error)
    !call MPI_COMM_SIZE(MPI_COMM_WORLD,nbprocs,error)
    !groupsize = 5

    call cart_create(topodims,error, main_comm)
    !call set_group_size(groupSize)
    ! Create meshes
    call discretisation_create(ncells(1),ncells(2),ncells(3),lengths(1),lengths(2),lengths(3))

    ! Init advection solver
    call advec_init(order,stab_coeff,dim_split=dim_split)

    ! get the local resolution (saved in scales global variable "N_proc")
    datashape = N_proc
    ! get offset (i.e. global index of the lowest point of the current subdomain == scales global var "coord")
    offset = coord * datashape

  end subroutine init_advection_solver

  !> To change velocity resolution
  !!    @param[in] Nx   = number of points along X
  !!    @param[in] Ny   = number of points along Y
  !!    @param[in] Nz   = number of points along Z
  !!    @param[in] formula   = interpolation formula to use ('lin', 'L4_4' ,'M4')
  subroutine init_multiscale(Nx, Ny, Nz, formula)

    integer, intent(in) :: Nx, Ny, Nz
    character(len=*), optional, intent(in)  ::  formula

    call discretisation_set_mesh_Velo(Nx, Ny, Nz)
    if(present(formula)) then
      call interpol_init(formula, .true.)
    else
      call interpol_init('L4_4', .true.)
    end if

  end subroutine init_multiscale

  !> Particular solver for the advection of a scalar
  !! @param[in] dt current time step
  !! @param[in] vx x component of the velocity used for advection
  !! @param[in] vy y component of the velocity used for advection
  !! @param[in] vz z component of the velocity used for advection
  !! @param[in,out] scal 3d scalar field which is advected
  subroutine solve_advection(dt,vx,vy,vz,scal)

    real(pk), intent(in) :: dt
    real(pk), dimension(:,:,:), intent(in) :: vx, vy, vz
    real(pk), dimension(size(vx,1),size(vx,2),size(vx,3)), intent(inout) :: scal
    !f2py real(pk) intent(in,out), depend(size(vx,1)) :: scal

    real(pk) :: t0

    t0 = MPI_Wtime()
    call advec_step(dt,vx,vy,vz,scal)
    !!print *, "inside ...", cart_rank, ":", MPI_Wtime()-t0
  end subroutine solve_advection

  !> Particular solver for the advection of a scalar
  !! @param[in] dt current time step
  !! @param[in] vx x component of the velocity used for advection
  !! @param[in] vy y component of the velocity used for advection
  !! @param[in] vz z component of the velocity used for advection
  !! @param[in,out] cx 3d scalar field which is advected
  !! @param[in,out] cy 3d scalar field which is advected
  !! @param[in,out] cz 3d scalar field which is advected
  subroutine solve_advection_vect(dt,vx,vy,vz,cx,cy,cz)

    real(pk), intent(in) :: dt
    real(pk), dimension(:,:,:), intent(in) :: vx, vy, vz
    real(pk), dimension(size(vx,1),size(vx,2),size(vx,3)), intent(inout) :: cx,cy,cz
    !f2py real(pk) intent(in,out), depend(size(vx,1)) :: cx
    !f2py real(pk) intent(in,out), depend(size(vx,1)) :: cy
    !f2py real(pk) intent(in,out), depend(size(vx,1)) :: cz

    real(8) :: t0

    t0 = MPI_Wtime()
    call advec_step_Vect(dt,vx,vy,vz,cx,cy,cz)
    !!print *, "inside ...", cart_rank, ":", MPI_Wtime()-t0
  end subroutine solve_advection_vect


  !> Solve advection equation - order 2 - with basic velocity interpolation
  !!    @param[in]        dt          = time step
  !!    @param[in]        Vx          = velocity along x (could be discretised on a bigger mesh then the scalar)
  !!    @param[in]        Vy          = velocity along y
  !!    @param[in]        Vz          = velocity along z
  !!    @param[in,out]    scal        = scalar field to advect
  subroutine solve_advection_inter_basic(dt, Vx, Vy, Vz, scal)

    ! Input/Output
    real(pk), intent(in)                        :: dt
    real(pk), dimension(:,:,:), intent(in)      :: Vx, Vy, Vz
    real(pk), dimension(:,:,:), intent(inout)   :: scal
    !f2py intent(in,out) :: scal

    call advec_step_Inter_basic(dt, Vx, Vy, Vz, scal)

  end subroutine solve_advection_inter_basic

  !> Solve advection equation - order 2 - with basic velocity interpolation
  !!    @param[in]        dt          = time step
  !!    @param[in]        Vx          = velocity along x (could be discretised on a bigger mesh then the scalar)
  !!    @param[in]        Vy          = velocity along y
  !!    @param[in]        Vz          = velocity along z
  !! @param[in,out] cx 3d scalar field which is advected
  !! @param[in,out] cy 3d scalar field which is advected
  !! @param[in,out] cz 3d scalar field which is advected
  subroutine solve_advection_inter_basic_vec(dt, Vx, Vy, Vz, cx, cy, cz)

    ! Input/Output
    real(pk), intent(in)                        :: dt
    real(pk), dimension(:,:,:), intent(in)      :: Vx, Vy, Vz
    real(pk), dimension(:,:,:), intent(inout)   :: cx,cy,cz
    !f2py intent(in,out) :: cx
    !f2py intent(in,out) :: cy
    !f2py intent(in,out) :: cz

    call advec_step_Inter_basic_Vect(dt, Vx, Vy, Vz, cx, cy, cz)

  end subroutine solve_advection_inter_basic_vec


end module scales2py
