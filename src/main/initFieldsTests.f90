!> Some functions to initialize fields to specific functions -> test purpose
module initFields

use client_data

implicit none

contains

  !> Set values for input field (omega)
  subroutine computeOmega3d(omega_x,omega_y,omega_z,step,refx,refy,refz,offset,lengths)
    
    real(mk),dimension(:,:,:) :: omega_x,omega_y,omega_z,refx,refy,refz
    real(mk),dimension(3) :: step
    real(mk) :: x,y,z,cx,cy,cz,cden,Lx,Ly,Lz
    integer(C_INTPTR_T) :: i,j,k
    integer(C_INTPTR_T),dimension(3),intent(in) :: offset
    real(mk),dimension(3),intent(in) ::lengths
    Lx = lengths(c_X)
    Ly = lengths(c_Y)
    Lz = lengths(c_Z)
    cx = 2*pi/Lx
    cy = 2*pi/Ly
    cz = 2*pi/Lz
    cden = 4*pi**2*(Ly**2*Lz**2+Lx**2*Lz**2+Lx**2*Ly**2)/(Lx**2*Ly**2*Lz**2)
    do k = 1,size(omega_x,3)
       z = (k-1+offset(c_Z))*step(c_Z)
       do j = 1, size(omega_x,2)
          y = (j-1+offset(c_Y))*step(c_Y)
          do i = 1, size(omega_x,1)
             x = (i-1+offset(c_X))*step(c_X)
             omega_x(i,j,k) = cden*(sin(x*cx)*sin(y*cy)*cos(z*cz))
             omega_y(i,j,k) = cden*(cos(x*cx)*sin(y*cy)*sin(z*cz))
             omega_z(i,j,k) = cden*(cos(x*cx)*cos(y*cy)*sin(z*cz))
             refx(i,j,k) = -2.*pi/Ly*(cos(x*cx)*sin(y*cy)*sin(z*cz))-2.*pi/Lz*(cos(x*cx)*sin(y*cy)*cos(z*cz))
             refy(i,j,k) = -2.*pi/Lz*(sin(x*cx)*sin(y*cy)*sin(z*cz))+2.*pi/Lx*(sin(x*cx)*cos(y*cy)*sin(z*cz))
             refz(i,j,k) = -2.*pi/Lx*(sin(x*cx)*sin(y*cy)*sin(z*cz))-2.*pi/Ly*(sin(x*cx)*cos(y*cy)*cos(z*cz))
          end do
       end do
    end do

  end subroutine computeOmega3d
    !> Set values for input field (omega)
  subroutine computeOmega3dC(omega_x,omega_y,omega_z,step,refx,refy,refz,offset,lengths)
    
    complex(mk),dimension(:,:,:) :: omega_x,omega_y,omega_z,refx,refy,refz
    real(mk),dimension(3) :: step
    real(mk) :: x,y,z,cx,cy,cz,cden,Lx,Ly,Lz
    integer(C_INTPTR_T) :: i,j,k
    integer(C_INTPTR_T),dimension(3),intent(in) :: offset
    real(mk),dimension(3),intent(in) ::lengths

    Lx = lengths(c_X)
    Ly = lengths(c_Y)
    Lz = lengths(c_Z)
    cx = 2*pi/Lx
    cy = 2*pi/Ly
    cz = 2*pi/Lz
    cden = 4*pi**2*(Ly**2*Lz**2+Lx**2*Lz**2+Lx**2*Ly**2)/(Lx**2*Ly**2*Lz**2)
    do k = 1,size(omega_x,3)
       z = (k-1+offset(c_Z))*step(c_Z)
       do j = 1, size(omega_x,2)
          y = (j-1+offset(c_Y))*step(c_Y)
          do i = 1, size(omega_x,1)
             x = (i-1+offset(c_X))*step(c_X)
             omega_x(i,j,k) = cden*(sin(x*cx)*sin(y*cy)*cos(z*cz))
             omega_y(i,j,k) = cden*(cos(x*cx)*sin(y*cy)*sin(z*cz))
             omega_z(i,j,k) = cden*(cos(x*cx)*cos(y*cy)*sin(z*cz))
             refx(i,j,k) = -2.*pi/Ly*(cos(x*cx)*sin(y*cy)*sin(z*cz))-2.*pi/Lz*(cos(x*cx)*sin(y*cy)*cos(z*cz))
             refy(i,j,k) = -2.*pi/Lz*(sin(x*cx)*sin(y*cy)*sin(z*cz))+2.*pi/Lx*(sin(x*cx)*cos(y*cy)*sin(z*cz))
             refz(i,j,k) = -2.*pi/Lx*(sin(x*cx)*sin(y*cy)*sin(z*cz))-2.*pi/Ly*(sin(x*cx)*cos(y*cy)*cos(z*cz))
          end do
       end do
    end do
  end subroutine computeOmega3dC
  
  !> Set values for input field (omega)
  !! Temporary function for test purpose
  subroutine computeOmega2D(omega,step,refx,refy,offset,lengths)
    
    real(mk),dimension(:,:) :: omega,refx,refy
    real(mk),dimension(2) :: step
    real(mk) :: x,y,Lx,Ly,cx,cy,coeff
    integer(C_INTPTR_T) :: i,j
    integer(C_INTPTR_T),dimension(2),intent(in) :: offset
    real(mk),dimension(2),intent(in) ::lengths
    Lx = lengths(c_X)
    Ly = lengths(c_Y)
    cx = 2*pi/Lx
    cy = 2*pi/Ly
    coeff = 4*pi**2*(1./Lx**2+1./Ly**2)
    
    do j = 1, size(omega,2)
       y = (j-1+offset(c_Y))*step(c_Y)
       do i = 1, size(omega,1)
          x = (i-1+offset(c_X))*step(c_X) ! warning : no offset for x in input (distribution over y)
          omega(i, j) = coeff*cos(x*cx)*sin(y*cy)
          refx(i,j) = 2.*pi/Ly*cos(x*cx)*cos(y*cy)
          refy(i,j) = 2.*pi/Lx*sin(x*cx)*sin(y*cy)
       end do
    end do

  end subroutine computeOmega2D
  !> Set values for input field (omega)
  !! Temporary function for test purpose
  subroutine computeOmega2DC(omega,step,refx,refy,offset,lengths)
    
    complex(mk),dimension(:,:) :: omega,refx,refy
    real(mk),dimension(2) :: step
    real(mk) :: x,y,Lx,Ly,cx,cy,coeff
    integer(C_INTPTR_T) :: i,j
    integer(C_INTPTR_T),dimension(2),intent(in) :: offset
    real(mk),dimension(2),intent(in) ::lengths
    Lx = lengths(c_X)
    Ly = lengths(c_Y)
    cx = 2*pi/Lx
    cy = 2*pi/Ly
    coeff = 4*pi**2*(1./Lx**2+1./Ly**2)
    do j = 1, size(omega,2)
       y = (j-1+offset(c_Y))*step(c_Y)
       do i = 1, size(omega,1)
          x = (i-1+offset(c_X))*step(c_X) ! warning : no offset for x in input (distribution over y)
          omega(i, j) = coeff*cos(x*cx)*sin(y*cy)
          refx(i,j) = 2.*pi/Ly*cos(x*cx)*cos(y*cy)
          refy(i,j) = 2.*pi/Lx*sin(x*cx)*sin(y*cy)
       end do
    end do
  end subroutine computeOmega2DC
 
end module initFields
