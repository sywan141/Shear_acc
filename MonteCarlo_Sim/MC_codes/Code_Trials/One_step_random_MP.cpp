#include <iostream>
#include <boost/multiprecision/cpp_dec_float.hpp>
#include <filesystem>
#include <fstream>
#include <yaml-cpp/yaml.h>
#include <random>
#include <omp.h>
#include <chrono>


namespace YAML {
    template <>
    struct convert<boost::multiprecision::cpp_dec_float_50> {
        static Node encode(const boost::multiprecision::cpp_dec_float_50& rhs) {
            return Node(rhs.str(50, std::ios::fixed)); // 转换为字符串，保留 50 位精度
        }
    
        static bool decode(const Node& node, boost::multiprecision::cpp_dec_float_50& rhs) {
            if (!node.IsScalar()) {
                return false;
            }
            try {
                rhs = boost::multiprecision::cpp_dec_float_50(node.as<std::string>());
                return true;
            } catch (...) {
                return false;
            }
        }
    };
    }

namespace mp = boost::multiprecision;
using cpp_dec_float_50 = mp::cpp_dec_float_50;

namespace fs = std::filesystem;

const cpp_dec_float_50 m_p = 1.67262192369e-24;  // proton mass (cgs)
const cpp_dec_float_50 m_e = 9.1093837015e-28;  // electron mass (cgs)
const cpp_dec_float_50 c = 2.99792458e10;       // speed of light (cgs)
const cpp_dec_float_50 e = 4.80320425e-10;      // electron charge (cgs)
const cpp_dec_float_50 pi = 3.14159265358979323846;

const cpp_dec_float_50 m_par = m_e;

// Load configuration file (.yaml)
YAML::Node readConfig(const std::string & configPath)
{
    try
    {
        return YAML::LoadFile(configPath);
    }
    catch(const YAML::Exception& e)
    {
        std::cerr << "Loading Error" << e.what() << std::endl;
        std::exit(1);
    }
}

// linspace
std::vector<cpp_dec_float_50> Linspace(cpp_dec_float_50 start, cpp_dec_float_50 end, int nums)
{
    std::vector<cpp_dec_float_50> results(nums);

    if (nums == 1) // only single element is needed
    {   
        std::vector<cpp_dec_float_50> results = {start};
        return results;
    }

    cpp_dec_float_50 L_step = (end - start)/ (nums-1);

    for (int i = 0; i < nums; ++i)
    {
        results[i] = start + i * L_step; 
    }
    return results;
}

// logspace
std::vector<cpp_dec_float_50> Logspace(cpp_dec_float_50 start, cpp_dec_float_50 end, int nums)
{
    std::vector<cpp_dec_float_50> results(nums);

    for (int i = 0; i < nums; ++i)
    {
        results[i] = mp::pow(cpp_dec_float_50(10), Linspace(start , end, nums)[i]); 
        //printf("%.2f\n", Linspace(start , end, nums)[i]);
    }
    //printf("%.2f\n", results[0]);
    return results;
}

// acceleration
cpp_dec_float_50 LorentzGamma(cpp_dec_float_50 beta1, cpp_dec_float_50 beta2, cpp_dec_float_50 costheta, cpp_dec_float_50 g_me, bool approx)
{
    cpp_dec_float_50 betaD = (beta2 - beta1)/(1 - beta1*beta2);
    cpp_dec_float_50 betae = mp::sqrt(1 - 1/ mp::pow(g_me, 2));
    cpp_dec_float_50 dGM = 1.0 / mp::sqrt(1 - mp::pow(betaD,2));
    cpp_dec_float_50 gme2;
    cpp_dec_float_50 Gamma1;

    if (approx == true) // use energy approximation
    {   
        Gamma1 = 1 / (1 - beta1 * beta1);
        cpp_dec_float_50 Dbeta = beta2 - beta1;
        betaD = (Gamma1 * Gamma1) * Dbeta *(1 + Gamma1 * Gamma1 * Dbeta * beta1); // 存在影响
        //gme2 = dGM * g_me * (1 - betae * betaD * costheta);  // 主要问题
        gme2 = g_me * (1 + 0.5 * (betaD * betaD) - betae * betaD * costheta); //
    }

    else
    {
        gme2 = dGM * g_me * (1 - betae * betaD * costheta);
    }
    return gme2;
}

// jet profile
cpp_dec_float_50 Beta_Dis(cpp_dec_float_50 r, cpp_dec_float_50 R_sh, cpp_dec_float_50 GM0, cpp_dec_float_50 eta, cpp_dec_float_50 beta_min)
{
    cpp_dec_float_50 beta_max = mp::sqrt(1-1/(GM0*GM0));
    cpp_dec_float_50 r1 = eta*R_sh;
    if (r < r1)
    {
        return beta_max;
    } else if (r > R_sh)
    {
        return 0.0;
    } else
    {
        return beta_max - (beta_max - beta_min)/(R_sh - r1)* (r - r1);
    }
}

// movement
std::tuple<cpp_dec_float_50,cpp_dec_float_50> movement_e(cpp_dec_float_50 gme, cpp_dec_float_50 costheta, cpp_dec_float_50 phi, cpp_dec_float_50 x, cpp_dec_float_50 y, cpp_dec_float_50 dt)
{
    cpp_dec_float_50 sintheta = mp::sqrt(1 - costheta*costheta);
    cpp_dec_float_50 u_cmv = mp::sqrt(1 - 1 / (gme*gme));

    cpp_dec_float_50 ux = u_cmv * sintheta * mp::cos(phi);
    cpp_dec_float_50 uy = u_cmv * sintheta * mp::sin(phi);

    x += ux * c * dt;
    y += uy * c * dt;

    return {x, y};
}

// movement with radius
cpp_dec_float_50 movement_e2 (cpp_dec_float_50 gme, cpp_dec_float_50 costheta, cpp_dec_float_50 alpha, cpp_dec_float_50 phi, cpp_dec_float_50 r0, cpp_dec_float_50 dt, bool mode)
{
    cpp_dec_float_50 sintheta = mp::sqrt( 1 - costheta * costheta);
    cpp_dec_float_50 u_cmv = mp::sqrt (1 - 1 / (gme * gme));

    cpp_dec_float_50 dx = u_cmv * sintheta * mp::cos(phi) *c *dt;
    cpp_dec_float_50 dy = u_cmv * sintheta * mp::sin(phi) *c *dt;
    cpp_dec_float_50 dr;
    if (mode == false)
    {
        dr = mp::sqrt ( (r0 * mp::cos(alpha) + dx) * (r0 * mp::cos(alpha) + dx) + (r0 * mp::sin(alpha) + dy) * (r0 * mp::sin(alpha) + dy)) - r0;
    }
    else
    {
        dr = dx * mp::cos(alpha);
    }
    
    return dr;
}

// gyro radius
cpp_dec_float_50 R_g (cpp_dec_float_50 gme, cpp_dec_float_50 B0)
{
    return gme * m_e * c *c / (e*B0);
}

// scattering timescale
cpp_dec_float_50 tau_calc (cpp_dec_float_50 gme, cpp_dec_float_50 B0, cpp_dec_float_50 q, cpp_dec_float_50 Lam_max, cpp_dec_float_50 xi)
{
    cpp_dec_float_50 rg = R_g(gme, B0);
    cpp_dec_float_50 sc_tau = mp::pow(rg , 2-q) * mp::pow(Lam_max , q-1)/ (c * xi);
    return sc_tau;
}

cpp_dec_float_50 Coeff_A ( cpp_dec_float_50 beta, cpp_dec_float_50 GM0, cpp_dec_float_50 eta, cpp_dec_float_50 R_sh)
{
    cpp_dec_float_50 Gamma_j = 1 / (mp::sqrt( 1- beta * beta));
    cpp_dec_float_50 beta0 = mp::sqrt( 1 - 1 / (GM0 * GM0));
    cpp_dec_float_50 Grad = beta0 / ((1 - eta) * R_sh);
    cpp_dec_float_50 A = ( Gamma_j * Gamma_j) * Grad * c;
    return A;
}

// define 
struct Allparticles
{
    std::vector<cpp_dec_float_50> G_res;
    std::vector<cpp_dec_float_50> costhetas;
    std::vector<cpp_dec_float_50> alphas;
    std::vector<cpp_dec_float_50> phis;

};

Allparticles Single_Par ( int K, unsigned long seed, cpp_dec_float_50 r0, cpp_dec_float_50 R_sh, cpp_dec_float_50 GM0, cpp_dec_float_50 eta, 
    cpp_dec_float_50 beta_min, cpp_dec_float_50 B0, cpp_dec_float_50 q, cpp_dec_float_50 Lam_max, cpp_dec_float_50 xi, 
    int N_bins, int MoveType, std::vector<cpp_dec_float_50> gmes, bool E_approx)
{   
    // use random seeds
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> dist (0.0, 1.0); // 使用 double 生成随机数，后转为 cpp_dec_float_50

    //initialization
    int Gamma_num = gmes.size();
    cpp_dec_float_50 beta_ini = Beta_Dis(r0, R_sh, GM0, eta, beta_min);
    Allparticles result;
    result.G_res.resize(gmes.size(), 0.0);
    result.costhetas.resize(gmes.size(), 0.0);
    result.alphas.resize(gmes.size(), 0.0);
    result.phis.resize(gmes.size(), 0.0);

    for (int i = 0; i < Gamma_num; ++i)
    {
        cpp_dec_float_50 gme_i = gmes[i];
        cpp_dec_float_50 gme0 = gme_i;
        cpp_dec_float_50 tau = tau_calc(gme0, B0, q ,Lam_max, xi);
        cpp_dec_float_50 costheta = cpp_dec_float_50(2.0 * dist(rng) - 1);
        cpp_dec_float_50 alpha = cpp_dec_float_50(2.0 * pi * dist(rng));
        cpp_dec_float_50 phi = cpp_dec_float_50(2.0 * pi * dist(rng));
        cpp_dec_float_50 dt = tau/N_bins;
        cpp_dec_float_50 r_tmp = r0;
        cpp_dec_float_50 beta_tmp;
        cpp_dec_float_50 x; cpp_dec_float_50 y;

        result.costhetas[i] = costheta;
        result.alphas[i] = alpha;
        result.phis[i] = phi;

        // time steps
        for (int Nth = 0; Nth < N_bins; ++ Nth)
        {
            if (MoveType == 1)
            {   
                x = r0 * mp::cos(alpha); y = r0 * mp::sin(alpha);
                //long double r_ini = std::sqrt(x * x + y * y);
                auto [renew_x, renew_y] = movement_e(gme_i , costheta , phi, x , y, dt);
                r_tmp = mp::sqrt(renew_x * renew_x + renew_y * renew_y);

                // Move out of range
                if (r_tmp > R_sh) 
                {
                    result.G_res[i] = std::numeric_limits<cpp_dec_float_50>::quiet_NaN();
                    std::cout << " moves out of the jet" << std::endl;
                    break;
                }

                beta_tmp = Beta_Dis(r_tmp, R_sh, GM0, eta, beta_min);
                costheta = cpp_dec_float_50(2.0 * dist(rng) - 1);
                alpha = cpp_dec_float_50(2.0 * pi * dist(rng));
                phi  = cpp_dec_float_50(2.0 * pi * dist(rng));
                x = r0 * mp::cos(alpha); y = r0 * mp::sin(alpha);
                r_tmp = r0;

            }
            else if (MoveType == 2) // no approximation
            {
                cpp_dec_float_50 dr = movement_e2 (gme_i, costheta, alpha, phi, r0, dt, false);
                r_tmp = r0 + dr;

                // Move out of range
                if (r_tmp > R_sh) 
                {
                    result.G_res[i] = std::numeric_limits<cpp_dec_float_50>::quiet_NaN();
                    std::cout << " moves out of the jet" << std::endl;
                    break;
                }

                beta_tmp = Beta_Dis(r_tmp, R_sh, GM0, eta, beta_min);
                costheta = cpp_dec_float_50(2.0 * dist(rng) - 1);
                alpha = cpp_dec_float_50(2.0 * pi * dist(rng));
                phi  = cpp_dec_float_50(2.0 * pi * dist(rng));
                r_tmp = r0;
            }

            else if (MoveType == 3) // with approximation
            {   
                //long double sintheta = std::sqrt (1 - costheta * costheta);
                //long double dx = std::sqrt(1 - 1/(gme0 * gme0)) * sintheta * std::cos(phi) *c *dt;
                cpp_dec_float_50 dr = movement_e2 (gme_i, costheta, alpha, phi, r0, dt, true);
                r_tmp = r0 + dr;

                // Move out of range
                if (r_tmp > R_sh) 
                {
                    result.G_res[i] = std::numeric_limits<cpp_dec_float_50>::quiet_NaN();
                    std::cout << " moves out of the jet" << std::endl;
                    break;
                }

                beta_tmp = Beta_Dis(r_tmp, R_sh, GM0, eta, beta_min);
                costheta = cpp_dec_float_50(2.0 * dist(rng) - 1);
                alpha = cpp_dec_float_50(2.0 * pi * dist(rng));
                phi  = cpp_dec_float_50(2.0 * pi * dist(rng));
                r_tmp = r0;

            }
            gme_i = LorentzGamma (beta_ini, beta_tmp, costheta, gme_i, E_approx);

        }
        result.G_res[i] = gme_i - gme0;

    }
    return result;

}

int main()
{   
    auto start_t = std::chrono::steady_clock::now();
    // load path
    std::string configPath = "/home/wsy/Acc_MC/MC_sim/paras.yaml";
    std::string outputDir = "/home/wsy/Acc_MC/MC_sim/codes/Code_Trials/cpp_random/";
    YAML::Node config = readConfig(configPath);

    // read files
    cpp_dec_float_50 R_sh = config["R_sh"].as<cpp_dec_float_50>();
    cpp_dec_float_50 GM0 = config["GM0"].as<cpp_dec_float_50>();
    cpp_dec_float_50 eta = config["eta"].as<cpp_dec_float_50>();
    cpp_dec_float_50 beta_min = config["beta_min"].as<cpp_dec_float_50>();
    cpp_dec_float_50 B0 = config["B0"].as<cpp_dec_float_50>();
    cpp_dec_float_50 xi = config["xi"].as<cpp_dec_float_50>();
    cpp_dec_float_50 Lam_max = config["Lam_max"].as<cpp_dec_float_50>();
    //cpp_dec_float_50 g_me0 = config["g_me0"].as<cpp_dec_float_50>();
    cpp_dec_float_50 r0 = config["r0"].as<cpp_dec_float_50>();
    cpp_dec_float_50 n_p = config["n_p"].as<cpp_dec_float_50>();

    int N_par = config["N_par"].as<int>();
    int N_time = config["N_time"].as<int>();
    int N_bins = config["N_bins"].as<int>();
    int Move_type = config["Type"].as<int>(); // select type of movement

    std::string jet_type = config["type"].as<std::string>();
    bool syn = config["SYN_flag"].as<bool>();
    bool SA = config["SA_flag"].as<bool>();
    bool Sh = config["Shear_flag"].as<bool>();
    bool ESC = config["ESC_flag"].as<bool>();
    bool E_approx = config["E_approx"].as<bool>(); // whether use approximations for energy
    bool Integrat = config["Integration"].as<bool>(); // whether use numerical integration

    cpp_dec_float_50 q;
    if (jet_type == "kolgv") 
    {
        q = cpp_dec_float_50("5.0") / cpp_dec_float_50("3.0");
    } 
    else if (jet_type == "Bohm") 
    {
        q = cpp_dec_float_50("1.0");
    } 
    else if (jet_type == "HS") 
    {
        q = cpp_dec_float_50("2.0");
    }
    else 
    {
        std::cerr << "Invalid jet type" << std::endl;
        return 1;
    }

    int gamma_num = 100; // energy numbers
    std::vector<cpp_dec_float_50> gmes = Logspace(cpp_dec_float_50(1), cpp_dec_float_50(9), gamma_num);

    std::vector<Allparticles> results (N_par); // initialize
    
    // generate seeds
    std::mt19937 seed_rng(1234);
    std::uniform_int_distribution<unsigned long> seed_dist(0, std::numeric_limits<unsigned long>::max());
    std::vector<unsigned long> seeds(N_par);
    // assign a seed for each particle
    for (int i = 0; i< N_par; ++i)
    {
        seeds[i] = seed_dist(seed_rng);
    }

    // multiprocess
    #pragma omp parallel for schedule(dynamic)
    for (int K = 0; K < N_par; ++K)
    {
        results[K] = Single_Par(K, seeds[K], r0, R_sh, GM0, eta, beta_min, B0, q, Lam_max, xi, 
            N_bins, Move_type, gmes, E_approx );
            #pragma omp critical
            {
                std::cout << "Particle" << K+1 << "finished !" << std::endl; 
            }
    }

    // save to text files
    if (fs::exists(outputDir))
    {
        fs::remove_all(outputDir);
    }
    fs::create_directories(outputDir);

    // initialize and open the files
    std::vector<std::string> filenames = {"Gammas.txt", "Costhetas.txt", "Alphas.txt", "Phis.txt"};
    std::vector<std::ofstream> outFiles;
    for (const auto& fname : filenames)
    {
        std::ofstream out (outputDir + fname);
        out << std::fixed << std::setprecision(50);
        if (!out.is_open()) 
        {
            std::cerr << "Error opening file: " << outputDir + fname << std::endl;
            return 1;
        }
        outFiles.push_back(std::move(out));
    }
    std::string outputFile = outputDir + "gmes.txt"; // save initial energy
    std::ofstream outFile(outputFile);
    outFile << std::fixed << std::setprecision(50); 

    // write data into the files
    for (int i = 0; i< N_par; ++i)
    {
        for (int j = 0; j < gamma_num; ++j)
        {
            outFiles[0] << results[i].G_res[j] << (j < gamma_num - 1 ? " " : "\n");
            outFiles[1] << results[i].costhetas[j] << (j < gamma_num - 1 ? " " : "\n");
            outFiles[2] << results[i].alphas[j] << (j < gamma_num - 1 ? " " : "\n");
            outFiles[3] << results[i].phis[j] << (j < gamma_num - 1 ? " " : "\n");
        }
    } 

    for (int k =0; k < gamma_num; ++k)
    {
        outFile << gmes[k] << (k < gamma_num - 1 ? " " : "\n"); // change rows
    }

    // close files
    for (auto& out : outFiles)
    {
        out.close();
    }
    outFile.close();

    // print running time
    auto end_t = std::chrono::steady_clock::now();
    double duration = std::chrono::duration<double>(end_t - start_t).count();
    std::cout << "Results saved! Running time: " << duration << "s" << std::endl;

    return 0;
}