!> Test program the fortran part of the HySoP library
program mainHySoP

use client_data
use poisson
use initFields
use vectorcalculus

implicit none

integer :: info
real(mk) :: start, end

!complex(mk), dimension(resolution(1),resolution(2)) :: omega,velocity_x,velocity_y
call MPI_Init(info)
call MPI_COMM_RANK(MPI_COMM_WORLD,rank,info)
call MPI_COMM_SIZE(MPI_COMM_WORLD,nbprocs,info)

!call test2D_r2c()

!call test2D_c2c()

start = MPI_WTIME()
call test3D_r2c()
print *, "time for basic version : ", MPI_WTIME() - start

!start = MPI_WTIME()
!call test3D_r2c_many()
!print *, "time for many version : ", MPI_WTIME() - start

!call test3D_c2c()

call MPI_Finalize(info)


contains
  subroutine test2D_r2c()

    integer, dimension(2),parameter :: resolution =(/65,65/)
    real(mk), dimension(:,:),pointer :: omega,velocity_x,velocity_y,refx,refy
    real(mk),dimension(2) :: lengths,step
    integer(C_INTPTR_T),dimension(2) :: nfft,offset
    real(mk) :: error
    logical :: ok
    integer, dimension(3) :: ghosts_v, ghosts_w

    !    call MPI_BARRIER(MPI_COMM_WORLD)
    if (rank==0) print *, " ======= Test 2D Poisson solver (r2c) for resolution ======= ", resolution

    lengths(:)=2*pi!!1.0
    step(:)=lengths(:)/(resolution(:)-1)
    ghosts_v(:) = 0
    ghosts_w(:) = 0
    !call init_r2c_2dBIS(resolution,lengths)

    call initPoissonSolver(2,resolution,lengths, MPI_COMM_WORLD)

    call getParamatersTopologyFFTW2d(nfft,offset)
    allocate(omega(nfft(c_X),nfft(c_Y)),velocity_x(nfft(c_X),nfft(c_Y)),velocity_y(nfft(c_X),nfft(c_Y)))
    allocate(refx(nfft(c_X),nfft(c_Y)),refy(nfft(c_X),nfft(c_Y)))
    call computeOmega2D(omega,step,refx,refy,offset,lengths)

    call solvePoisson2D(omega,velocity_x,velocity_y, ghosts_w, ghosts_v)
    call cleanPoissonSolver2D()

    ok = iscloseto_2d(refx,velocity_x,tolerance,step,error)
    if(rank==0) write(*,'(a,L2,3f10.4)') 'Solver convergence (x): ', ok, error
    ok = iscloseto_2d(refy,velocity_y,tolerance,step,error)
    if(rank==0)  write(*,'(a,L2,3f10.4)') 'Solver convergence (y): ', ok, error

    if (rank==0) print *, " ======= End of Test 2D Poisson ======= "

  end subroutine test2D_r2c

  subroutine test2D_c2c()

    integer, dimension(2),parameter :: resolution =(/65,65/)
    complex(mk), dimension(:,:),pointer :: omega,velocity_x,velocity_y,refx,refy
    real(mk),dimension(2) :: lengths,step
    integer(C_INTPTR_T),dimension(2) :: nfft,offset
    real(mk) :: error
    logical :: ok

    if (rank==0) print *, " ======= Test 2D Poisson solver (c2c) for resolution ======= ", resolution

    lengths(:)=2*pi!!1.0
    step(:)=lengths(:)/(resolution(:)-1)

!    call MPI_BARRIER(MPI_COMM_WORLD)

    call initPoissonSolverC(2,resolution,lengths,MPI_COMM_WORLD)

    call getParamatersTopologyFFTW2d(nfft,offset)

    allocate(omega(nfft(c_X),nfft(c_Y)),velocity_x(nfft(c_X),nfft(c_Y)),velocity_y(nfft(c_X),nfft(c_Y)))
    allocate(refx(nfft(c_X),nfft(c_Y)),refy(nfft(c_X),nfft(c_Y)))

    call computeOmega2DC(omega,step,refx,refy,offset,lengths)

    call solvePoisson2DC(omega,velocity_x,velocity_y)
    call cleanPoissonSolver2D()

    ok = iscloseto_2dc(refx,velocity_x,tolerance,step,error)
    if(rank==0) write(*,'(a,L2,3f10.4)') 'Solver convergence (x): ', ok, error
    ok = iscloseto_2dc(refy,velocity_y,tolerance,step,error)
    if(rank==0)  write(*,'(a,L2,3f10.4)') 'Solver convergence (y): ', ok, error

    deallocate(omega,velocity_x,velocity_y,refx,refy)

    if (rank==0) print *, " ======= End of Test 2D Poisson ======= "


  end subroutine test2D_c2c

  subroutine test3D_r2c()

    integer, dimension(3),parameter :: resolution =(/256,256,256/)
    real(mk), dimension(:,:,:),pointer :: omega_x,omega_y,omega_z,velocity_x,velocity_y,velocity_z,refx,refy,refz
    real(mk),dimension(3) :: lengths,step
    integer, dimension(3) :: ghosts_v, ghosts_w
    integer(C_INTPTR_T),dimension(3) :: nfft,offset
    real(mk) :: error,start
    logical :: ok

    if (rank==0) print *, " ======= Test 3D Poisson (r2c) solver for resolution  ", resolution

    lengths(:)=2*pi!!1.0
    step(:)=lengths(:)/(resolution(:)-1)
    ghosts_v(:) = 0
    ghosts_w(:) = 0
    !    call MPI_BARRIER(MPI_COMM_WORLD)

    call initPoissonSolver(3,resolution,lengths,MPI_COMM_WORLD)

    call getParamatersTopologyFFTW3d(nfft,offset)

    allocate(omega_x(nfft(c_X),nfft(c_Y),nfft(c_Z)),omega_y(nfft(c_X),nfft(c_Y),nfft(c_Z)),omega_z(nfft(c_X),nfft(c_Y),nfft(c_Z)))
    allocate(velocity_x(nfft(c_X),nfft(c_Y),nfft(c_Z)),velocity_y(nfft(c_X),nfft(c_Y),nfft(c_Z)),&
         velocity_z(nfft(c_X),nfft(c_Y),nfft(c_Z)))
    allocate(refx(nfft(c_X),nfft(c_Y),nfft(c_Z)),refy(nfft(c_X),nfft(c_Y),nfft(c_Z)),refz(nfft(c_X),nfft(c_Y),nfft(c_Z)))

    call computeOmega3D(omega_x,omega_y,omega_z,step,refx,refy,refz,offset,lengths)

    start = MPI_Wtime()
    call solvePoisson3D(omega_x,omega_y,omega_z,velocity_x,velocity_y,velocity_z, ghosts_w, ghosts_v)
    print *, "resolution time : ", MPI_Wtime()-start
    call cleanPoissonSolver3D()

    ok = iscloseto_3d(refx,velocity_x,tolerance,step,error)
    if(rank==0) write(*,'(a,L2,3f10.4)') 'Solver convergence (x): ',ok, error
    ok = iscloseto_3d(refy,velocity_y,tolerance,step,error)
    if(rank==0)  write(*,'(a,L2,3f10.4)') 'Solver convergence (y): ',ok, error
    ok = iscloseto_3d(refz,velocity_z,tolerance,step,error)
    if(rank==0)  write(*,'(a,L2,3f10.4)') 'Solver convergence (z): ', ok,error

    deallocate(omega_x,omega_y,omega_z,velocity_x,velocity_y,velocity_z,refx,refy,refz)

    if (rank==0) print *, " ======= End of Test 3D Poisson ======= "

  end subroutine test3D_r2c

  subroutine test3D_r2c_many()

    integer, dimension(3),parameter :: resolution =(/65,65,65/)

    real(mk), dimension(:,:,:),pointer :: omega_x,omega_y,omega_z,velocity_x,velocity_y,velocity_z,refx,refy,refz
    real(mk),dimension(3) :: lengths,step
    integer(C_INTPTR_T),dimension(3) :: nfft,offset
    real(mk) :: error
    logical :: ok

    if (rank==0) print *, " ======= Test 3D Poisson (r2c many) solver for resolution ======= ", resolution

    lengths(:)=2*pi!!1.0
    step(:)=lengths(:)/(resolution(:)-1)

    !    call MPI_BARRIER(MPI_COMM_WORLD)

    call init_r2c_3d_many(resolution,lengths)

    call getParamatersTopologyFFTW3d(nfft,offset)

    allocate(omega_x(nfft(c_X),nfft(c_Y),nfft(c_Z)),omega_y(nfft(c_X),nfft(c_Y),nfft(c_Z)),omega_z(nfft(c_X),nfft(c_Y),nfft(c_Z)))
    allocate(velocity_x(nfft(c_X),nfft(c_Y),nfft(c_Z)),velocity_y(nfft(c_X),nfft(c_Y),nfft(c_Z)),&
         velocity_z(nfft(c_X),nfft(c_Y),nfft(c_Z)))
    allocate(refx(nfft(c_X),nfft(c_Y),nfft(c_Z)),refy(nfft(c_X),nfft(c_Y),nfft(c_Z)),refz(nfft(c_X),nfft(c_Y),nfft(c_Z)))

    call computeOmega3D(omega_x,omega_y,omega_z,step,refx,refy,refz,offset,lengths)

    call solvePoisson3D_many(omega_x,omega_y,omega_z,velocity_x,velocity_y,velocity_z)
    call cleanFFTW_3d()

    ok = iscloseto_3d(refx,velocity_x,tolerance,step,error)
    if(rank==0) write(*,'(a,L2,3f10.4)') 'Solver convergence (x): ',ok, error
    ok = iscloseto_3d(refy,velocity_y,tolerance,step,error)
    if(rank==0)  write(*,'(a,L2,3f10.4)') 'Solver convergence (y): ',ok, error
    ok = iscloseto_3d(refz,velocity_z,tolerance,step,error)
    if(rank==0)  write(*,'(a,L2,3f10.4)') 'Solver convergence (z): ', ok,error

    deallocate(omega_x,omega_y,omega_z,velocity_x,velocity_y,velocity_z,refx,refy,refz)

    if (rank==0) print *, " ======= End of Test 3D Poisson ======= "

  end subroutine test3D_r2c_many

  subroutine test3D_c2c()

    integer, dimension(3),parameter :: resolution =(/65,65,65/)
    complex(mk), dimension(:,:,:),pointer :: omega_x,omega_y,omega_z,velocity_x,velocity_y,velocity_z,refx,refy,refz
    real(mk),dimension(3) :: lengths,step
    integer(C_INTPTR_T),dimension(3) :: nfft,offset
    real(mk) :: error
    logical :: ok

    if (rank==0) print *, " ======= Test 3D Poisson (c2c) solver for resolution ======= ", resolution

    lengths(:)=2*pi!!1.0
    step(:)=lengths(:)/(resolution(:)-1)

!    call MPI_BARRIER(MPI_COMM_WORLD)

    call initPoissonSolverC(3,resolution,lengths,MPI_COMM_WORLD)

    call getParamatersTopologyFFTW3d(nfft,offset)

    allocate(omega_x(nfft(c_X),nfft(c_Y),nfft(c_Z)),omega_y(nfft(c_X),nfft(c_Y),nfft(c_Z)),omega_z(nfft(c_X),nfft(c_Y),nfft(c_Z)))
    allocate(velocity_x(nfft(c_X),nfft(c_Y),nfft(c_Z)),velocity_y(nfft(c_X),nfft(c_Y),nfft(c_Z)),&
         velocity_z(nfft(c_X),nfft(c_Y),nfft(c_Z)))
    allocate(refx(nfft(c_X),nfft(c_Y),nfft(c_Z)),refy(nfft(c_X),nfft(c_Y),nfft(c_Z)),refz(nfft(c_X),nfft(c_Y),nfft(c_Z)))

    call computeOmega3DC(omega_x,omega_y,omega_z,step,refx,refy,refz,offset,lengths)

    call solvePoisson3DC(omega_x,omega_y,omega_z,velocity_x,velocity_y,velocity_z)
    call cleanPoissonSolver3D()

    ok = iscloseto_3dc(refx,velocity_x,tolerance,step,error)
    if(rank==0) write(*,'(a,L2,3f10.4)') 'Solver convergence (x): ', ok,error
    ok = iscloseto_3dc(refy,velocity_y,tolerance,step,error)
    if(rank==0)  write(*,'(a,L2,3f10.4)') 'Solver convergence (y): ', ok,error
    ok = iscloseto_3dc(refz,velocity_z,tolerance,step,error)
    if(rank==0)  write(*,'(a,L2,3f10.4)') 'Solver convergence (z): ', ok,error

    deallocate(omega_x,omega_y,omega_z,velocity_x,velocity_y,velocity_z,refx,refy,refz)

    if (rank==0) print *, " ======= End of Test 3D Poisson ======= "


  end subroutine test3D_c2c



end program mainHySoP
