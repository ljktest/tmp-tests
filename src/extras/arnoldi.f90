!=========================================================================
!Computation of global modes using a time stepping (snapshots) technique
!=========================================================================
!=========================================================================
!Reads snapshots, constructs the Hessenberg matrix
!and computes the eigen-values/eigen-functions
!=========================================================================
module arnoldi

  use client_data

  implicit none

contains

!======================
subroutine arnoldi3d(Mu,ncli,nt,nfp,nmodes,Tps)
  implicit none
  integer, intent(in) :: ncli ! number of snapshot
  integer, intent(in) :: nt  ! total number of points per snapshot
  integer, intent(in) :: nmodes  ! number of desired modes
  integer, intent(in) :: nfp ! number of desired eigen functions
  real(mk), intent(in) :: Tps ! sampling time step
  real(mk), dimension(:,:), intent(inout) :: Mu ! snapshots

  real(mk), dimension(:,:), allocatable :: un  ! orthonomalized Krylov basis
  real(mk), dimension(:,:), allocatable :: Hessenberg ! Hessenberg matrix
  complex(mk), dimension(:), allocatable :: VP ! eigenvalues
  complex(mk), dimension(:,:), allocatable :: FP, FP_J ! eigen functions
  real(mk), dimension(:), allocatable :: reslog, res ! residuals
  integer, dimension(:), allocatable :: t ! sorting array
  real(mk), dimension(:), allocatable :: tab !
  real(mk)  :: norm, prod, error !

  integer :: i,j,k,nmax

  allocate(un(nt,ncli),Hessenberg(ncli,ncli-1),VP(ncli-1),FP(ncli-1,ncli-1),FP_J(nt,ncli))

  nmax=0
  VP=(0.,0.); FP=(0.,0.); FP_J=(0.,0.)
  Hessenberg=0.

  !==================================
  !   Arnoldi method
  !==================================

  norm = dot_product(Mu(:,1),Mu(:,1))
  norm = sqrt(norm)
  un(:,1) = Mu(:,1)/norm  ! first normalized vector u1

  do j=2,ncli ! construct a normalized base u2... un
    un(:,j)=Mu(:,j) ! w=Mu_j (We have Mu_j=U(:,j+1))
    do i=1,j-1
      Hessenberg(i,j-1)=dot_product(un(:,i),un(:,j))
      un(:,j)=un(:,j)-un(:,i)*Hessenberg(i,j-1)
    enddo

    norm = dot_product(un(:,j),un(:,j))
    Hessenberg(j,j-1) = sqrt(norm)

    un(:,j) = un(:,j)/Hessenberg(j,j-1)! normalization

  enddo

!  do i=1,nt
!    print *, 'Krylov basis:', un(i,:)
!  enddo

  do i=1,ncli-1
    print *, 'Hessenberg matrix:', Hessenberg(i,:)
  enddo


!Check orthonormalization
!==================================

  print *,'check ortho'

  prod=0.
  do i=1,ncli
    do j=1,ncli
      prod=dot_product(un(:,j),un(:,i))
      if (abs(prod).gt.1e-14) then
        print *,i,j,abs(prod)
      endif
    enddo
  enddo


!Eigen-values and Eigen-functions related to Hessenberg matrix
! +
!Eigen-values related to Jacobian matrix ==> spectra
!==============================================================

  open(unit=10, file='spectrum.dat')
  open(unit=11, file='spectrum_log.dat')

call spectrum(Hessenberg(1:ncli-1,:),ncli-1,VP,FP)

  do i=1,ncli-1
     write(10,*) dble(VP(i)), aimag(VP(i))
     write(11,*) dble(log(VP(i)))/Tps, ATAN(aimag(VP(i)/DBLE(VP(i))))/Tps
  enddo
  close(10)
  close(11)

!Eigen-functions related to Jacobian matrix
!==========================================
  FP_J(1:nt,1:ncli-1)=matmul(un(1:nt,1:ncli-1),FP(1:ncli-1,1:ncli-1))
!  do i=1,nt
!    print *, 'FP_J', (FP_J(i,j),j=1,ncli-1)
!  enddo

!Residual calculation with respect to each mode
!==============================================

  allocate(res(ncli-1),reslog(ncli-1))
  error = Hessenberg(ncli,ncli-1)
  print *,'last Hess',Hessenberg(ncli,ncli-1)

  do i=1,ncli-1
    res(i)   = abs(FP(ncli-1,i))*error
    reslog(i)=-log10(res(i))
    print *,'residual',reslog(i),res(i)
  enddo


!Modes are sorted with respect to residual
!==========================================
  allocate(t(ncli-1))

  do i=1,ncli-1
    t(i)=i
  enddo

  call sort(reslog,ncli-1,t)

  open(unit=201,file='spectrum_res.dat')
  write(201,*)'VARIABLES ="WR","WI","RES"'

  do i=1,nmodes
    write(201,100) dble(log(VP(t(i))))/Tps,&
                    ATAN(aimag(VP(t(i))/DBLE(VP(t(i)))))/Tps,&
                     res(t(i))
  enddo
  close(201)
!
!Write the associated eigen functions
!====================================
!  allocate(tab(nfp))
!
!  open(unit=107, file='table.dat')
!  open(unit=108, file='spectrum_sorted.dat')
!
!  do i=1,nfp
!!    call ecriture(FP_J(:,t(h)))
!    write(108,*) ATAN(aimag(VP(t(i))/DBLE(VP(t(i)))))/Tps,&
!                      dble(log(VP(t(i))))/Tps
!  enddo
!  close(108)
!
100   format (5(2x,e19.13))

end subroutine arnoldi3d

!===================
!Spectrum subroutine
!===================
subroutine spectrum(A,n,VP,VR)
  implicit none
  integer              :: INFO
  integer              :: n,LWORK
  real(mk), dimension(n,n) :: A
  real(mk), dimension(:), allocatable :: RWORK
  complex(mk), dimension(1,n) :: VL
  complex(mk), dimension(n,n) :: VR
  complex(mk), dimension(:), allocatable :: WORK
  complex(mk), dimension(n):: VP

  LWORK=4*n
  allocate(WORK(LWORK),RWORK(2*n))
  call ZGEEV('N','V', n, A*(1.,0.), n, VP, VL, 1, VR, n,&
       WORK, LWORK, RWORK, INFO )
  print *, 'VP', VP

end subroutine spectrum

!==================
!Sorting subroutine
!==================
subroutine sort(t,n,ind)
  implicit none
  integer                  :: i, j, n, tp1
  real(mk), dimension(1:n) :: t
  real(mk)                 :: temp
  integer, dimension(1:n)  :: ind

  do i=1,n
     do j=i+1,n
        if ((t(i))<(t(j))) then

           temp=t(i)
           tp1=ind(i)

           t(i)=t(j)
           ind(i)=ind(j)

           t(j)=temp
           ind(j)=tp1

        endif
     enddo
  enddo

  return

end subroutine sort

end module arnoldi
