#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <random>
#include <filesystem>
#include <yaml-cpp/yaml.h>
#include <omp.h>

namespace fs = std::filesystem;

// 定义物理常量
const double m_p = 1.67262192369e-24;  // 质子质量 (g)
const double m_e = 9.1093837015e-28;  // 电子质量 (g)
const double c = 2.99792458e10;       // 光速 (cm/s)
const double e = 4.80320425e-10;      // 电子电荷 (esu)
const double pi = 3.14159265358979323846;

const double m_par = m_e;

// 读取配置文件
YAML::Node readConfig(const std::string& configPath) {
    try 
    {
        return YAML::LoadFile(configPath);
    } catch (const YAML::Exception& e)
    {
        std::cerr << "Error loading YAML file: " << e.what() << std::endl;
        std::exit(1);
    }
}

// jet速度谱
double beta_dis(double r, double R_sh, double GM0, double eta, double beta_min) {
    double beta_max = std::sqrt(1 - 1 / (GM0 * GM0));
    double r1 = eta * R_sh;
    if (r < r1) 
    {
        return beta_max;
    } else if (r > R_sh) 
    {
        return 0.0;
    } else 
    {
        return beta_max - (beta_max - beta_min) / (R_sh - r1) * (r - r1);
    }
}

// 获取任意向量天顶角和方位角
std::pair<double, double> Get_Angles(const std::vector<double>& u) 
{
    double ux = u[0], uy = u[1], uz = u[2];
    double norm = std::sqrt(ux * ux + uy * uy + uz * uz);
    if (norm == 0) 
    {
        throw std::invalid_argument("The input should be a non-zero vector");
    }
    double theta = std::acos(uz / norm);
    double phi = std::atan2(uy, ux);
    if (phi < 0) phi += 2 * pi;
    return {theta, phi};
}

// 将速度转换到新的共动系中
std::vector<double> Vel_Shear(double beta1, double beta2, const std::vector<double>& u_cmv)
 {
    double ux = u_cmv[0], uy = u_cmv[1], uz = u_cmv[2];
    double dbeta = (beta2 - beta1) / (1 - beta1 * beta2);
    double gm_dbeta = 1 / std::sqrt(1 - dbeta * dbeta);
    double ux_prime = ux / (gm_dbeta * (1 - dbeta * uz));
    double uy_prime = uy / (gm_dbeta * (1 - dbeta * uz));
    double uz_prime = (uz - dbeta) / (1 - dbeta * uz);
    return {ux_prime, uy_prime, uz_prime};
}

// 将时间转换到观测者系中
double LorentzT(double beta, double dz, double dt) 
{
    double GM = 1 / std::sqrt(1 - beta * beta);
    return GM * (dt + beta / c * dz);
}

// 位移函数
std::tuple<double, double, double, double> movement_e(const std::vector<double>& u_cmv, double beta, double dt, double x, double y, double z)
 {
    double ux = u_cmv[0], uy = u_cmv[1], uz = u_cmv[2];
    double dx = ux * c * dt;
    double dy = uy * c * dt;
    double dz = 1 / std::sqrt(1 - beta * beta) * c * (beta + uz) * dt;
    double dz_jet = uz * c * dt; //共动系下的轴向位移
    x += dx;
    y += dy;
    z += dz;
    return {x, y, z, dz_jet};
}

// 坐标变换矩阵
std::vector<std::vector<double>> Rot_mat(const std::string& direct, double angle) 
{
    double cosa = std::cos(angle);
    double sina = std::sin(angle);
    if (direct == "x") 
    {
        return {{1.0, 0.0, 0.0},
                {0.0, cosa, -sina},
                {0.0, sina, cosa}};
    } else if (direct == "y") 
    {
        return {{cosa, 0.0, sina},
                {0.0, 1.0, 0.0},
                {-sina, 0.0, cosa}};
    } else if (direct == "z") 
    {
        return {{cosa, -sina, 0.0},
                {sina, cosa, 0.0},
                {0.0, 0.0, 1.0}};
    }
    return {};
}

// 矩阵乘法
std::vector<double> matrixMultiply(const std::vector<std::vector<double>>& A, const std::vector<double>& B) 
{
    std::vector<double> result(A.size(), 0.0);
    for (size_t i = 0; i < A.size(); ++i)   
    {
        for (size_t j = 0; j < B.size(); ++j)
        {
            result[i] += A[i][j] * B[j];
        }
    }
    return result;
}

// 各向同性散射，生成随机方向的单位速度矢量
std::vector<double> isotropic_scatter(const std::vector<double>& u_cmv) 
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    double theta = std::acos(2 * dis(gen) - 1);
    double phi = 2 * pi * dis(gen);
    std::vector<double> u_out = 
    {
        std::sin(theta) * std::cos(phi),
        std::sin(theta) * std::sin(phi),
        std::cos(theta)
    };
    double u_norm = std::sqrt(u_cmv[0] * u_cmv[0] + u_cmv[1] * u_cmv[1] + u_cmv[2] * u_cmv[2]);
    for (double& val : u_out) 
    {
        val *= u_norm;
    }
    return u_out;
}

// 随机加速情况，在Alfven波参考系中进行各向同性散射
std::tuple<std::vector<double>, double, double> SA_scatter(const std::vector<double>& u_cmv, double ba) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> dis(0.0, 1.0);
    std::vector<double> k = {dis(gen), dis(gen), dis(gen)};
    double k_norm = std::sqrt(k[0] * k[0] + k[1] * k[1] + k[2] * k[2]);
    for (double& val : k) 
    {
        val /= k_norm;
    }
    auto [theta, phi] = Get_Angles(k);
    std::vector<double> u_wav = matrixMultiply(Rot_mat("y", theta), matrixMultiply(Rot_mat("z", phi), u_cmv));
    double uk_x = u_wav[0], uk_y = u_wav[1], uk_z = u_wav[2];
    double gamma_k = 1 / std::sqrt(1 - ba * ba);
    double theta_in = Get_Angles(u_wav).first;

    double ukk_x = uk_x / (gamma_k * (1 - ba * uk_z));
    double ukk_y = uk_y / (gamma_k * (1 - ba * uk_z));
    double ukk_z = (uk_z - ba) / (1 - ba * uk_z);
    std::vector<double> u_kk = {ukk_x, ukk_y, ukk_z};
    std::vector<double> u_wav_out = isotropic_scatter(u_kk);
    double uk_x_out = u_wav_out[0], uk_y_out = u_wav_out[1], uk_z_out = u_wav_out[2];
    double theta_out = Get_Angles(u_wav_out).first;

    double ukx_out = uk_x_out / (gamma_k * (1 + ba * uk_z_out));
    double uky_out = uk_y_out / (gamma_k * (1 + ba * uk_z_out));
    double ukz_out = (uk_z_out + ba) / (1 + ba * uk_z_out);
    std::vector<double> uk_out = {ukx_out, uky_out, ukz_out};
    std::vector<double> uj_out = matrixMultiply(Rot_mat("z", -phi), matrixMultiply(Rot_mat("y", -theta), uk_out));

    return {uj_out, theta_in, theta_out};
}

// 能量变换
double LorentzGamma(double beta1, double beta2, double theta, double g_me)
{
    double costheta = std::cos(theta);
    double dbeta = (beta2 - beta1) / (1 - beta1 * beta2);
    if (dbeta == 0) 
    {
        return g_me;
    }
    double betae = std::sqrt(1 - 1 / (g_me * g_me));
    double dGM = 1.0 / std::sqrt(1 - dbeta * dbeta);
    return dGM * g_me * (1 - betae * dbeta * costheta);
}

// 镜面反射，返回反射后的二维速度
std::vector<double> Relect(const std::vector<double>& uxy, const std::vector<double>& k) 
{
    double u_norm = std::sqrt(uxy[0] * uxy[0] + uxy[1] * uxy[1]);
    double k_norm = std::sqrt(k[0] * k[0] + k[1] * k[1]);
    std::vector<double> norm_k = {k[0] / k_norm, k[1] / k_norm};
    double costheta = (uxy[0] * k[0] + uxy[1] * k[1]) / (u_norm * k_norm);
    std::vector<double> uy_re = { -u_norm * costheta * norm_k[0], -u_norm * costheta * norm_k[1] };
    std::vector<double> ux_re = {uxy[0] + uy_re[0], uxy[1] + uy_re[1]};
    return {ux_re[0] + uy_re[0], ux_re[1] + uy_re[1]};
}

// 单粒子模拟
std::tuple<std::vector<double>, std::vector<double>, std::vector<double>, std::vector<double>, std::vector<double>, std::vector<double>, std::vector<double>, std::vector<double>> Single_Par
(double R_sh, double GM0, double eta, double beta_min, double B0, double xi, double Lam_max, double g_me0, double r0, double n_p, double q, 
    int N_time, int N_bins, 
    bool syn, bool SA, bool Sh, bool ESC) 

{
    double rg = std::sqrt(g_me0 * g_me0 - 1) * m_e * c * c / e / B0;
    double ba = B0 / std::sqrt(B0 * B0 + 4 * pi * n_p * m_p * c * c);
    double Dpp = xi * ba * ba / (1 - ba * ba) * g_me0 * g_me0 * c / std::pow(rg, 2 - q) / std::pow(Lam_max, q - 1);
    double tacc = g_me0 * g_me0 / Dpp;
    double tau = std::pow(rg, 2 - q) * std::pow(Lam_max, q - 1) / (c * xi);

    double theta_prev = 0;
    double phi_prev = 0;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(eta, 1.0);
    r0 = dis(gen) * R_sh;

    //初始化
    double t_j = 0;
    double t_o = 0;
    double dt = tau / N_bins;
    double g_me = g_me0;
    double r = r0;
    double x = r0, y = 0, z = 0;
    double beta1 = beta_dis(r0, R_sh, GM0, eta, beta_min);
    double beta2 = beta_dis(r0, R_sh, GM0, eta, beta_min);
    double beta_ini = 0;
    double gme_obs = LorentzGamma(beta1, beta_ini, -1, g_me0);

    std::vector<double> gme_jetL(N_time + 2, 0.0);
    std::vector<double> r_jetL(N_time + 2, 0.0);
    std::vector<double> gme_obsL(N_time + 2, 0.0);
    std::vector<double> t_jetL(N_time + 2, 0.0);
    std::vector<double> t_obsL(N_time + 2, 0.0);
    std::vector<double> x_jetL(N_time + 2, 0.0);
    std::vector<double> y_jetL(N_time + 2, 0.0);
    std::vector<double> z_jetL(N_time + 2, 0.0);

    gme_jetL[0] = g_me;
    r_jetL[0] = r;
    gme_obsL[0] = gme_obs;
    t_jetL[0] = t_j;
    t_obsL[0] = t_o;
    x_jetL[0] = r0;
    y_jetL[0] = 0;
    z_jetL[0] = 0;

    int N_count = N_time;
    while (N_count >= 0)
     {
        std::uniform_real_distribution<> dis(0.0, 1.0);
        double poss = dis(gen);
        if (poss > 1 - std::exp(-dt / tau))
         {
            double u_cmv0 = std::sqrt(1 - 1 / (g_me * g_me)); //速度模
            std::vector<double> u_cmv = {u_cmv0 * std::sin(theta_prev) * std::cos(phi_prev),
                                         u_cmv0 * std::sin(theta_prev) * std::sin(phi_prev),
                                         u_cmv0 * std::cos(theta_prev)};
            u_cmv = Vel_Shear(beta1, beta2, u_cmv); //坐标变换

            if (Sh) //剪切加速效应
            {
                double theta = Get_Angles(u_cmv).first;
                g_me = LorentzGamma(beta1, beta2, theta, g_me);
            }

            beta1 = beta_dis(r, R_sh, GM0, eta, beta_min);
            auto [new_x, new_y, new_z, dz_jet] = movement_e(u_cmv, beta2, dt, x, y, z);
            x = new_x;
            y = new_y;
            z = new_z;
            r = std::sqrt(x * x + y * y);

            if (syn) 
            {
                g_me -= 1.1e-15 * g_me * g_me * B0 * B0 / m_par / (c * c) * dt;
            }

            if (r > R_sh) 
            {
                if (ESC) 
                {
                    std::fill(gme_jetL.begin() + N_time - N_count + 1, gme_jetL.end(), -1);
                    std::fill(r_jetL.begin() + N_time - N_count + 1, r_jetL.end(), -1);
                    std::fill(gme_obsL.begin() + N_time - N_count + 1, gme_obsL.end(), -1);
                    std::fill(t_jetL.begin() + N_time - N_count + 1, t_jetL.end(), -1);
                    std::fill(t_obsL.begin() + N_time - N_count + 1, t_obsL.end(), -1);
                    std::fill(x_jetL.begin() + N_time - N_count + 1, x_jetL.end(), 10 * R_sh);
                    std::fill(y_jetL.begin() + N_time - N_count + 1, y_jetL.end(), 10 * R_sh);
                    std::fill(z_jetL.begin() + N_time - N_count + 1, z_jetL.end(), -1);
                    break;
                } 
                else 
                {
                    std::vector<double> ux = {u_cmv[0], u_cmv[1]};
                    std::vector<double> k = {x, y};
                    std::vector<double> uxy_re = Relect(ux, k);
                    u_cmv = {uxy_re[0], uxy_re[1], u_cmv[2]};
                    double Ang = std::atan2(y, x);

                    r = R_sh;
                    x = r * std::cos(Ang);
                    y = r * std::sin(Ang);
                }
            }

            // 更新参数
            t_j += dt;
            t_o += LorentzT(beta2, dz_jet, dt);
            rg = std::sqrt(g_me * g_me - 1) * m_e * c * c / e / B0;
            tau = std::pow(rg, 2 - q) * std::pow(Lam_max, q - 1) / (c * xi);
            gme_obs = LorentzGamma(beta2, beta_ini, -1, g_me);
            beta2 = beta_dis(r, R_sh, GM0, eta, beta_min);
            auto [theta, phi] = Get_Angles(u_cmv);
            theta_prev = theta;
            phi_prev = phi;

            t_jetL[N_time - N_count + 1] = t_j;
            t_obsL[N_time - N_count + 1] = t_o;
            gme_jetL[N_time - N_count + 1] = g_me;
            r_jetL[N_time - N_count + 1] = r;
            gme_obsL[N_time - N_count + 1] = gme_obs;
            x_jetL[N_time - N_count + 1] = x;
            y_jetL[N_time - N_count + 1] = y;
            z_jetL[N_time - N_count + 1] = z;

            N_count--;
            continue;
        }

        //散射分支
        double u_cmv0 = std::sqrt(1 - 1 / (g_me * g_me));
        std::vector<double> u_cmv = {u_cmv0 * std::sin(theta_prev) * std::cos(phi_prev),
                                     u_cmv0 * std::sin(theta_prev) * std::sin(phi_prev),
                                     u_cmv0 * std::cos(theta_prev)};
        u_cmv = Vel_Shear(beta1, beta2, u_cmv);

        if (Sh) 
        {
            double theta = Get_Angles(u_cmv).first;
            g_me = LorentzGamma(beta1, beta2, theta, g_me);
        }

        if (SA) 
        {
            auto [uj_out, theta_in, theta_out] = SA_scatter(u_cmv, ba);
            u_cmv = uj_out;
            g_me = LorentzGamma(0, ba, theta_in, g_me);
            g_me = LorentzGamma(ba, 0, theta_out, g_me);
            auto [theta, phi] = Get_Angles(u_cmv);
            theta_prev = theta;
            phi_prev = phi;
        } 
        else 
        {
            u_cmv = isotropic_scatter(u_cmv);
            auto [theta, phi] = Get_Angles(u_cmv);
            theta_prev = theta;
            phi_prev = phi;
        }

        if (syn) 
        {
            g_me -= 1.1e-15 * g_me * g_me * B0 * B0 / m_par / (c * c) * dt;
        }

        beta1 = beta_dis(r, R_sh, GM0, eta, beta_min);
        auto [new_x, new_y, new_z, dz_jet] = movement_e(u_cmv, beta2, dt, x, y, z);
        x = new_x;
        y = new_y;
        z = new_z;
        r = std::sqrt(x * x + y * y);

        if (r > R_sh) 
        {
            if (ESC) 
            {
                std::fill(gme_jetL.begin() + N_time - N_count + 1, gme_jetL.end(), -1);
                std::fill(r_jetL.begin() + N_time - N_count + 1, r_jetL.end(), -1);
                std::fill(gme_obsL.begin() + N_time - N_count + 1, gme_obsL.end(), -1);
                std::fill(t_jetL.begin() + N_time - N_count + 1, t_jetL.end(), -1);
                std::fill(t_obsL.begin() + N_time - N_count + 1, t_obsL.end(), -1);
                std::fill(x_jetL.begin() + N_time - N_count + 1, x_jetL.end(), 10 * R_sh);
                std::fill(y_jetL.begin() + N_time - N_count + 1, y_jetL.end(), 10 * R_sh);
                std::fill(z_jetL.begin() + N_time - N_count + 1, z_jetL.end(), -1);
                break;
            } 
            else 
            {
                std::vector<double> ux = {u_cmv[0], u_cmv[1]};
                std::vector<double> k = {x, y};
                std::vector<double> uxy_re = Relect(ux, k);
                u_cmv = {uxy_re[0], uxy_re[1], u_cmv[2]};
                double Ang = std::atan2(y, x);

                r = R_sh;
                x = r * std::cos(Ang);
                y = r * std::sin(Ang);
            }
        }

        //更新参数
        t_j += dt;
        t_o += LorentzT(beta2, dz_jet, dt);
        rg = std::sqrt(g_me * g_me - 1) * m_e * c * c / e / B0;
        tau = std::pow(rg, 2 - q) * std::pow(Lam_max, q - 1) / (c * xi);
        gme_obs = LorentzGamma(beta2, beta_ini, -1, g_me);
        beta2 = beta_dis(r, R_sh, GM0, eta, beta_min);

        t_jetL[N_time - N_count + 1] = t_j;
        t_obsL[N_time - N_count + 1] = t_o;
        gme_jetL[N_time - N_count + 1] = g_me;
        r_jetL[N_time - N_count + 1] = r;
        gme_obsL[N_time - N_count + 1] = gme_obs;
        x_jetL[N_time - N_count + 1] = x;
        y_jetL[N_time - N_count + 1] = y;
        z_jetL[N_time - N_count + 1] = z;

        N_count--;
    }

    return {t_jetL, t_obsL, gme_jetL, gme_obsL, r_jetL, x_jetL, y_jetL, z_jetL};
}

// 保存数组到txt文件
void writeToTXT(const std::vector<std::vector<double>>& array, const std::string& filePath) {
    std::ofstream file(filePath);
    if (file.is_open()) 
    {
        for (const auto& row : array) 
        {
            for (size_t i = 0; i < row.size(); ++i)
             {
                if (i > 0) 
                {
                    file << " ";
                }
                file << row[i];
            }
            file << "\n";
        }
        file.close();
    } 
    else 
    {
        std::cerr << "Unable to open file: " << filePath << std::endl;
    }
}
    
//主函数
int main() 
{   
    //加载路径
    std::string configPath = "/home/wsy/Acc_MC/MC_sim/paras.yaml";
    std::string outputDir = "/home/wsy/Acc_MC/Results/trial_Rotation_RW_cpp/";
    YAML::Node config = readConfig(configPath);

    //读取参数文件
    double R_sh = config["R_sh"].as<double>();
    double GM0 = config["GM0"].as<double>();
    double eta = config["eta"].as<double>();
    double beta_min = config["beta_min"].as<double>();
    double B0 = config["B0"].as<double>();
    double xi = config["xi"].as<double>();
    double Lam_max = config["Lam_max"].as<double>();
    double g_me0 = config["g_me0"].as<double>();
    double r0 = config["r0"].as<double>();
    double n_p = config["n_p"].as<double>();

    int N_par = config["N_par"].as<int>();
    int N_time = config["N_time"].as<int>();
    int N_bins = config["N_bins"].as<int>();

    std::string jet_type = config["type"].as<std::string>();
    bool syn = config["SYN_flag"].as<bool>();
    bool SA = config["SA_flag"].as<bool>();
    bool Sh = config["Shear_flag"].as<bool>();
    bool ESC = config["ESC_flag"].as<bool>();

    double q;
    if (jet_type == "kolgv") 
    {
        q = 5.0 / 3.0;
    } 
    else if (jet_type == "Bohm") 
    {
        q = 1.0;
    } 
    else if (jet_type == "HS") 
    {
        q = 2.0;
    }
    else 
    {
        std::cerr << "Invalid jet type" << std::endl;
        return 1;
    }

    //若已存在则删除
    if (fs::exists(outputDir)) 
    {
        fs::remove_all(outputDir);
    }
    fs::create_directories(outputDir);
    
    //初始化
    std::vector<std::vector<double>> t_jetL;
    std::vector<std::vector<double>> t_obsL;
    std::vector<std::vector<double>> gme_jetL;
    std::vector<std::vector<double>> gme_obsL;
    std::vector<std::vector<double>> r_jetL;
    std::vector<std::vector<double>> x_jetL;
    std::vector<std::vector<double>> y_jetL;
    std::vector<std::vector<double>> z_jetL;

    //OMP多线程计算
    #pragma omp parallel for num_threads(64)
    for (int i = 0; i < N_par; ++i) 
    {
        int thread_id = omp_get_thread_num();
        std::cout << "Thread " << thread_id << " is processing particle " << i << "\n" << std::endl;
        auto [t_jet, t_obs, gme_jet, gme_obs, r_jet, x_jet, y_jet, z_jet] = Single_Par(R_sh, GM0, eta, beta_min, B0, xi, Lam_max, g_me0, r0, n_p, q, N_time, N_bins, syn, SA, Sh, ESC);
        #pragma omp critical
        {
            t_jetL.push_back(t_jet);
            t_obsL.push_back(t_obs);
            gme_jetL.push_back(gme_jet);
            gme_obsL.push_back(gme_obs);
            r_jetL.push_back(r_jet);
            x_jetL.push_back(x_jet);
            y_jetL.push_back(y_jet);
            z_jetL.push_back(z_jet);
        }
    }

    //结果保存到csv文件内
    writeToTXT(t_jetL, outputDir + "t_jetL.txt");
    writeToTXT(t_obsL, outputDir + "t_obsL.txt");
    writeToTXT(gme_jetL, outputDir + "gme_jetL.txt");
    writeToTXT(r_jetL, outputDir + "r_jetL.txt");
    writeToTXT(gme_obsL, outputDir + "gme_obsL.txt");
    writeToTXT(x_jetL, outputDir + "x_jetL.txt");
    writeToTXT(y_jetL, outputDir + "y_jetL.txt");
    writeToTXT(z_jetL, outputDir + "z_jetL.txt");

    return 0;
}   