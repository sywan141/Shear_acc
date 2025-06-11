program ShearAcc
    use, intrinsic::iso_fortran_env, only: real128, int64 ! high precision
    implicit none

    ! constants
    real(real128), parameter :: m_p = 1.67262192369e-24_real128   ! proton mass (cgs)
    real(real128), parameter :: m_e = 9.1093837015e-28_real128    ! electron mass (cgs)
    real(real128), parameter :: c = 2.99792458e10_real128         ! speed of light (cgs)
    real(real128), parameter :: e_charge = 4.80320425e-10_real128 ! electron q (cgs)
    real(real128), parameter :: pi = 4.0_real128 * atan(1.0_real128) ! pi with high precision
    real(real128), parameter :: nan = transfer(z'7FF0000000000001', 1.0_real128) ! NaN value

    ! variables, should be defined before the programme runs
    integer :: i, j ,k, l, counts, ios
    integer :: gamma_num, Num_steps, Move_type
    real(real128) :: R_sh, GM0, eta, beta_min, B0, xi, Lam_max, r0 , q
    logical :: E_approx, Integrate
    character (len=20):: jet_type

    ! arrays
    real(real128), allocatable :: Gamma_list(:), Costheta_list(:), Alpha_list(:), Phi_list(:) ! 可变数组
    real(real128), allocatable :: res(:,:) 

    ! temporary variables
    real(real128) :: costheta, alpha, phi, sintheta
    real(real128) :: g_me, gme0, beta_ini, beta_tmp, C_A, tau, betae
    real(real128) :: x, y, renew_x, renew_y, dr, dt, r_tmp

    ! time counts
    integer(int64) :: start_count, end_count, count_rate
    real(real128) :: duration

    ! Initialize parameters
    R_sh = 1e20_real128
    GM0 = 1.1_real128
    eta = 0.0_real128
    beta_min = 0.0_real128
    B0 = 3.0e-6_real128
    xi = 0.1_real128
    Lam_max = 1.0e18_real128
    r0 = 1e16_real128
    Move_type = 1
    E_approx = .false.
    Integrate = .false.
    jet_type = "Bohm"

    if (jet_type == 'kolgv') then
        q = 5.0_real128/3.0_real128
    else if (jet_type == 'Bohm') then
        q = 1.0_real128
    else if (jet_type == 'HS') then
        q = 2.0_real128
    else 
        write (*,*) "No turbulence matched"
        stop
    end if

    gamma_num = 60 ! number of energy
    Num_steps = 60 ! number of grids


    ! allocate memories
    allocate(Gamma_list(gamma_num))
    allocate(Costheta_list(Num_steps))
    allocate(Alpha_list(Num_steps))
    allocate(Phi_list(Num_steps))
    allocate(res(Num_steps**3, gamma_num))

    ! create grids
    Gamma_list = logspace(1.0_real128, 9.0_real128, gamma_num)
    Costheta_list = linspace(-1.0_real128, 1.0_real128, Num_steps)
    Alpha_list = linspace(0.0_real128, 2.0_real128*pi, Num_steps)
    Phi_list = linspace(0.0_real128, 2.0_real128*pi, Num_steps)

    ! main loop
    write(*,*) " Calculation begins"
    call system_clock(start_count, count_rate)

    do i = 1, gamma_num
    counts = 1
    gme0 = Gamma_list(i)
    dt = tau_calc(gme0, B0,q, Lam_max, xi)

    do j = 1, Num_steps
    costheta = Costheta_list(j)

    do k = 1, Num_steps
    alpha = Alpha_list(k)

    do l = 1, Num_steps
    phi = Phi_list(l)
    beta_ini = Beta_Dis(r0, R_sh, GM0, eta, beta_min)
        ! integration method
        if (Integrate) then
            C_A = Coeff_A(beta_ini, GM0, eta, R_sh)
            tau = tau_calc(gme0, B0, q, Lam_max, xi)
            sintheta = sqrt(1.0_real128 - costheta**2)
            betae = sqrt(1.0_real128 - 1.0_real128/ (gme0**2))

            g_me = gme0 * (1.0_real128 + &
                            0.5_real128 * C_A**2 * betae**2 * tau**2 * sintheta**2 * cos(alpha)**2 - &
                            C_A * betae**2 * tau * sintheta * costheta * cos(alpha) - &
                            C_A**2 * beta_ini * betae**3 * tau * sintheta**2 * costheta * cos(alpha)**2)
        else
            select case (Move_type)
            case (1)                    ! coordinate movement
                x = r0 * cos(alpha)
                y = r0 * sin(alpha)
                call movement_e (gme0, costheta, phi, x, y, dt, renew_x, renew_y)
                r_tmp = sqrt (renew_x**2 + renew_y**2)

                ! check if the electron exceeds
                if (r_tmp > R_sh) then
                res(counts, i) = nan
                counts = counts + 1
                print *, "粒子超出喷流范围"
                cycle
                end if

                beta_tmp = Beta_Dis(r_tmp, R_sh, GM0, eta, beta_min)

            case (2)                    ! radi movement
                dr = movement_e2 (gme0, costheta, alpha, phi, r0, dt, .false.)
                r_tmp = r0 + dr
                ! check if the electron exceeds
                if (r_tmp > R_sh) then
                res(counts, i) = nan
                counts = counts + 1
                print *, "粒子超出喷流范围"
                cycle
                end if
                
                beta_tmp = Beta_Dis(r_tmp, R_sh, GM0, eta, beta_min)

            case (3)                   ! use approximation
                dr = movement_e2 (gme0, costheta, alpha, phi, r0, dt, .true.)
                r_tmp = r0 + dr
                ! check if the electron exceeds
                if (r_tmp > R_sh) then
                res(counts, i) = nan
                counts = counts + 1
                print *, "粒子超出喷流范围"
                cycle
                end if
                
                beta_tmp = Beta_Dis(r_tmp, R_sh, GM0, eta, beta_min)
            
            case default
            write(*,*) "Error, unknown move type"
            stop
            end select

            g_me = LorentzGamma(beta_ini, beta_tmp, costheta, gme0, E_approx)
        end if

        ! save results
        res(counts, i) = g_me - gme0
        counts = counts + 1
    end do
    end do
    end do
    write(*,*) "This energy finishes !"
    end do

    ! save results and calculate time
    call save_results("results_sol.txt", res, Gamma_list, gamma_num, Num_steps**3)
    call system_clock(end_count, count_rate)
    duration = real(end_count - start_count, real128) / (real(count_rate, real128))
    write (*,*) "Calculation finishes ! Time consumed:", duration, "s"
    ! free memories
    deallocate(Gamma_list, Costheta_list, Alpha_list, Phi_list, res)

! functions and subroutines, 函数和子程序内部都需要声明变量类型
contains
    ! 创建线性间隔数组
    function linspace(start, end, n) result(arr)
        real(real128), intent(in) :: start, end
        integer, intent(in) :: n
        real(real128) :: arr(n)
        real(real128) :: step
        integer :: i
        
        if (n == 1) then
            arr(1) = start
            return
        end if
        
        step = (end - start) / (n - 1)
        do i = 1, n
            arr(i) = start + (i - 1) * step
        end do
    end function linspace

    ! 对数间隔数组
    function logspace(start, end, n) result(arr)
        implicit none
        real(real128), intent(in) :: start, end  !声明为输入
        integer, intent(in) :: n
        real(real128) :: arr(n) ! 固定数组
        real(real128) :: step
        integer :: i

        if (n == 1) then
            arr(1) = 10.0_real128**start
            return
        end if
        
        step = (end - start) / (n - 1)
        do i = 1, n
            arr(i) = 10.0_real128**(start + (i - 1) * step)
        end do
    end function logspace

    ! 洛伦兹变换计算
    function LorentzGamma(beta1, beta2, costheta, gme, approx) result(gme2)
        real(real128), intent(in) :: beta1, beta2, costheta, gme
        logical, intent(in) :: approx
        real(real128) :: gme2, betaD, betae, dGM, Gamma1, Dbeta
        
        betaD = (beta2 - beta1) / (1.0_real128 - beta1 * beta2)
        betae = sqrt(1.0_real128 - 1.0_real128 / gme**2)
        
        if (approx) then
            ! 能量近似公式
            Gamma1 = 1.0_real128 / sqrt(1.0_real128 - beta1**2)
            Dbeta = beta2 - beta1
            betaD = (Gamma1**2) * Dbeta * (1.0_real128 + Gamma1**2 * beta1 * Dbeta)
            gme2 = gme * (1.0_real128 + 0.5_real128 * betaD**2 - betae * betaD * costheta)
        else
            ! 完整相对论公式
            dGM = 1.0_real128 / sqrt(1.0_real128 - betaD**2)
            gme2 = dGM * gme * (1.0_real128 - betae * betaD * costheta)
        end if
    end function LorentzGamma

    ! jet profile
    function Beta_Dis(r, R_sh, GM0, eta, beta_min) result(beta)
        real(real128), intent(in) :: r, R_sh, GM0, eta, beta_min
        real(real128) :: beta, beta_max, r1
        
        beta_max = sqrt(1.0_real128 - 1.0_real128 / GM0**2)
        r1 = eta * R_sh
        
        if (r < r1) then
            beta = beta_max
        else if (r > R_sh) then
            beta = 0.0_real128
        else
            beta = beta_max - (beta_max - beta_min) / (R_sh - r1) * (r - r1)
        end if
    end function Beta_Dis

    ! 二维坐标子程序
    subroutine movement_e (gme, costheta, phi,x,y,dt, xnew, ynew)
        real(real128), intent(in) :: gme, costheta, phi, x, y, dt
        real(real128), intent(out) :: xnew, ynew
        real(real128) :: sintheta, u_cmv, ux, uy

        sintheta = sqrt(1.0_real128 - costheta**2)
        u_cmv = sqrt(1.0_real128 - 1.0_real128 / gme**2)

        ux = u_cmv * sintheta * cos(phi)
        uy = u_cmv * sintheta * sin(phi)

        xnew = x + ux* c* dt
        ynew = y + uy* c* dt
    end subroutine movement_e

    ! 径向运动计算 (简化模型)
    function movement_e2(gme, costheta, alpha, phi, r0, dt, approx) result(dr)
        real(real128), intent(in) :: gme, costheta, alpha, phi, r0, dt
        logical, intent(in) :: approx
        real(real128) :: dr, sintheta, u_cmv, dx, dy
        
        sintheta = sqrt(1.0_real128 - costheta**2)
        u_cmv = sqrt(1.0_real128 - 1.0_real128 / gme**2)
        
        dx = u_cmv * sintheta * cos(phi) * c * dt
        dy = u_cmv * sintheta * sin(phi) * c * dt
        
        if (.not. approx) then
            ! 完整模型: 计算实际位移
            dr = sqrt((r0 * cos(alpha) + dx)**2 + (r0 * sin(alpha) + dy)**2) - r0
        else
            ! 近似模型: 仅考虑径向分量
            dr = dx * cos(alpha)
        end if
    end function movement_e2

    ! gyro radius
    function R_g(gme, B0) result(rg)
        real(real128), intent(in) :: gme, B0
        real(real128) :: rg
        
        rg = gme * m_e * c**2 / (e_charge * B0)
    end function R_g

    ! scattering timescale
    function tau_calc(gme, B0, q, Lam_max, xi) result(sc_tau)
        real(real128), intent(in) :: gme, B0, q, Lam_max, xi
        real(real128) :: sc_tau, rg
        
        rg = R_g(gme, B0)
        sc_tau = rg**(2.0_real128 - q) * Lam_max**(q - 1.0_real128) / (c * xi)
    end function tau_calc

    ! coefficient
    function Coeff_A(beta, GM0, eta, R_sh) result(A)
        real(real128), intent(in) :: beta, GM0, eta, R_sh
        real(real128) :: A, Gamma_j, beta0, Grad
        
        Gamma_j = 1.0_real128 / sqrt(1.0_real128 - beta**2)
        beta0 = sqrt(1.0_real128 - 1.0_real128 / GM0**2)
        Grad = beta0 / ((1.0_real128 - eta) * R_sh)
        A = (Gamma_j**2) * Grad * c
    end function Coeff_A

    ! Save results
    subroutine save_results (filename, res, Gamma_list, gamma_num, res_size)
        character(len=*), intent(in) :: filename
        real(real128), intent(in) :: res(:,:), Gamma_list(:)
        integer, intent(in) :: gamma_num, res_size
        integer :: fileunit, i, j, k, ios

        open(newunit=fileunit, file=filename, status='replace', action='write', iostat = ios)
        if (ios /= 0) then
            write (*,*) "错误: 无法创建结果文件"
            return
        end if
        
        ! 写入计算结果
        do i = 1, res_size
            do j = 1, gamma_num
                write(fileunit, '(G0.20)', advance='no') res(i, j)
                if (j < gamma_num) write(fileunit, '(" ")', advance='no')
            end do
            write(fileunit, *) ! 默认换行
        end do
        
        ! 写入初始能量值
        do k = 1, gamma_num
            write(fileunit, '(G0.20)', advance='no') Gamma_list(k)
            if (k < gamma_num) write(fileunit, '(" ")', advance='no')
        end do
        write(fileunit, *) ! 默认换行
        
        close(fileunit)
        print *, "Results saved: ", trim(filename)
    end subroutine save_results

end program ShearAcc


    
