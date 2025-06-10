#include <iostream>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <yaml-cpp/yaml.h>
#include <random>
#include <omp.h>
#include <boost/multiprecision/cpp_dec_float.hpp>

namespace fs = std::filesystem;

const long double m_p = 1.67262192369e-24L;  // proton mass (cgs)
const long double m_e = 9.1093837015e-28L;  // electron mass (cgs)
const long double c = 2.99792458e10L;       // speed of light (cgs)
const long double e = 4.80320425e-10L;      // electron charge (cgs)
const long double pi = 3.14159265358979323846L;

const long double m_par = m_e;


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


//linspace
std::vector<long double> Linspace(long double start, long double end, int nums)
{
    std::vector<long double> results(nums);

    if (nums == 1) // only single element is needed
    {   
        std::vector<long double> results = {start};
        return results;
    }

    long double L_step = (end - start)/ (nums-1);

    for (int i = 0; i < nums; ++i)
    {
        results[i] = start + i * L_step; 
    }
    return results;
}


// logspace
std::vector<long double> Logspace(long double start, long double end, int nums)
{
    std::vector<long double> results(nums);

    for (int i = 0; i < nums; ++i)
    {
        results[i] = pow(10, Linspace(start , end, nums)[i]); 
        //printf("%.2f\n", Linspace(start , end, nums)[i]);
    }
    //printf("%.2f\n", results[0]);
    return results;
}


// acceleration
long double LorentzGamma(long double beta1, long double beta2, long double costheta, long double g_me, bool approx)
{
    long double betaD = (beta2 - beta1)/(1 - beta1*beta2);
    long double betae = std::sqrt(1 - 1/ pow(g_me, 2));
    long double dGM = 1.0L / std::sqrt(1 - pow(betaD,2));
    long double gme2;
    long double Gamma1;

    if (approx == true) // use energy approximation
    {   
        Gamma1 = 1 / (1 - beta1 * beta1);
        long double Dbeta = beta2 - beta1;
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
long double Beta_Dis(long double r, long double R_sh, long double GM0, long double eta, long double beta_min)
{
    long double beta_max = std::sqrt(1-1/(GM0*GM0));
    long double r1 = eta*R_sh;
    if (r < r1)
    {
        return beta_max;
    } else if (r > R_sh)
    {
        return 0.0L;
    } else
    {
        return beta_max - (beta_max - beta_min)/(R_sh - r1)* (r - r1);
    }
}


// movement
std::tuple<long double,long double> movement_e(long double gme, long double costheta, long double phi, long double x, long double y, long double dt)
{
    long double sintheta = std::sqrt(1 - costheta*costheta);
    long double u_cmv = std::sqrt(1 - 1 / (gme*gme));

    long double ux = u_cmv * sintheta * std::cos(phi);
    long double uy = u_cmv * sintheta * std::sin(phi);

    x += ux * c * dt;
    y += uy * c * dt;

    return {x, y};
}

// movement with radius
double movement_e2 (long double gme, long double costheta, long double alpha, long double phi, long double r0, long double dt, bool mode)
{
    long double sintheta = std::sqrt( 1 - costheta * costheta);
    long double u_cmv = std::sqrt (1 - 1 / (gme * gme));

    long double dx = u_cmv * sintheta * std::cos(phi) *c *dt;
    long double dy = u_cmv * sintheta * std::sin(phi) *c *dt;
    long double dr;
    if (mode == false)
    {
        dr = std::sqrt ( (r0 * std::cos(alpha) + dx) * (r0 * std::cos(alpha) + dx) + (r0 * std::sin(alpha) + dy) * (r0 * std::sin(alpha) + dy)) - r0;
    }
    else
    {
        dr = dx * std::cos(alpha);
    }
    
    return dr;
}


// gyro radius
long double R_g (long double gme, long double B0)
{
    return gme * m_e * c *c / (e*B0);
}


// scattering timescale
long double tau_calc (long double gme, long double B0, long double q, long double Lam_max, long double xi)
{
    long double rg = R_g(gme, B0);
    long double sc_tau = pow(rg , 2-q) * pow(Lam_max , q-1)/ (c * xi);
    return sc_tau;
}


long double Coeff_A ( long double beta, long double GM0, long double eta, long double R_sh)
{
    long double Gamma_j = 1 / (std::sqrt( 1- beta * beta));
    long double beta0 = std::sqrt( 1 - 1 / (GM0 * GM0));
    long double Grad = beta0 / ((1 - eta) * R_sh);
    long double A = ( Gamma_j * Gamma_j) * Grad * c;
    return A;
}


// define 
struct Allparticles
{
    std::vector<long double> G_res;
    std::vector<long double> costhetas;
    std::vector<long double> alphas;
    std::vector<long double> phis;

};

Allparticles Single_Par ( int K, unsigned long seed, long double r0, long double R_sh, long double GM0, long double eta, 
    long double beta_min, long double B0, long double q, long double Lam_max, long double xi, 
    int N_bins, int MoveType, std::vector<long double> gme_list, bool E_approx)
{   
    // use random seeds
    std::mt19937 rng(seed);
    std::uniform_real_distribution <long double> dist (0.0L, 1.0L);

    //initialization
    int Gamma_num = gme_list.size();
    long double beta_ini = Beta_Dis(r0, R_sh, GM0, eta, beta_min);
    Allparticles result;
    result.G_res.resize(Gamma_num, 0.0L);
    result.costhetas.resize(Gamma_num, 0.0L);
    result.alphas.resize(Gamma_num, 0.0L);
    result.phis.resize(Gamma_num, 0.0L);

    for (int i = 0; i < Gamma_num; ++i)
    {
        long double gme_i = gme_list[i];
        long double gme2 = gme_i;
        long double tau = tau_calc(gme_i, B0, q ,Lam_max, xi);
        long double costheta = 2.0L * dist(rng) - 1;
        long double alpha = 2.0L * pi * dist(rng);
        long double phi = 2.0L * pi * dist(rng);
        long double dt = tau/N_bins;
        long double r_tmp = r0;
        long double beta_tmp;
        long double x; long double y;

        result.costhetas[i] = costheta;
        result.alphas[i] = alpha;
        result.phis[i] = phi;

        // time steps
        for (int Nth = 0; Nth < N_bins; ++ Nth)
        {
            if (MoveType == 1)
            {   
                x = r0 * std::cos(alpha); y = r0 * std::sin(alpha);
                //long double r_ini = std::sqrt(x * x + y * y);
                auto [renew_x, renew_y] = movement_e(gme_i , costheta , phi, x , y, dt);
                r_tmp = std::sqrt(renew_x * renew_x + renew_y * renew_y);

                // Move out of range
                if (r_tmp > R_sh) 
                {
                    result.G_res[i] = std::numeric_limits<long double>::quiet_NaN();
                    std::cout << " moves out of the jet" << std::endl;
                    break;
                }

                beta_tmp = Beta_Dis(r_tmp, R_sh, GM0, eta, beta_min);
                costheta = 2.0L * dist(rng) - 1;
                alpha = 2.0L * pi * dist(rng);
                phi  = 2.0L * pi * dist(rng);
                x = r0 * std::cos(alpha); y = r0 * std::sin(alpha);
                r_tmp = r0;

            }
            else if (MoveType == 2) // no approximation
            {
                double dr = movement_e2 (gme_i, costheta, alpha, phi, r0, dt, false);
                r_tmp = r0 + dr;

                // Move out of range
                if (r_tmp > R_sh) 
                {
                    result.G_res[i] = std::numeric_limits<long double>::quiet_NaN();
                    std::cout << " moves out of the jet" << std::endl;
                    break;
                }

                beta_tmp = Beta_Dis(r_tmp, R_sh, GM0, eta, beta_min);
                costheta = 2.0L * dist(rng) - 1;
                alpha = 2.0L * pi * dist(rng);
                phi  =2.0L * pi * dist(rng);
                r_tmp = r0;
            }

            else if (MoveType == 3) // with approximation
            {   
                //long double sintheta = std::sqrt (1 - costheta * costheta);
                //long double dx = std::sqrt(1 - 1/(gme0 * gme0)) * sintheta * std::cos(phi) *c *dt;
                double dr = movement_e2 (gme_i, costheta, alpha, phi, r0, dt, true);
                r_tmp = r0 + dr;

                // Move out of range
                if (r_tmp > R_sh) 
                {
                    result.G_res[i] = std::numeric_limits<long double>::quiet_NaN();
                    std::cout << " moves out of the jet" << std::endl;
                    break;
                }

                beta_tmp = Beta_Dis(r_tmp, R_sh, GM0, eta, beta_min);
                costheta = 2.0L * dist(rng) - 1;
                alpha = 2.0L * pi * dist(rng);
                phi  =2.0L * pi * dist(rng);
                r_tmp = r0;

            }
            gme2 = LorentzGamma (beta_ini, beta_tmp, costheta, gme2, E_approx);

        }

        result.G_res[i] = gme2 - gme_i;

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
    long double R_sh = config["R_sh"].as<long double>();
    long double GM0 = config["GM0"].as<long double>();
    long double eta = config["eta"].as<long double>();
    long double beta_min = config["beta_min"].as<long double>();
    long double B0 = config["B0"].as<long double>();
    long double xi = config["xi"].as<long double>();
    long double Lam_max = config["Lam_max"].as<long double>();
    //long double g_me0 = config["g_me0"].as<long double>();
    long double r0 = config["r0"].as<long double>();
    long double n_p = config["n_p"].as<long double>();

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

    long double q;
    if (jet_type == "kolgv") 
    {
        q = 5.0L / 3.0L;
    } 
    else if (jet_type == "Bohm") 
    {
        q = 1.0L;
    } 
    else if (jet_type == "HS") 
    {
        q = 2.0L;
    }
    else 
    {
        std::cerr << "Invalid jet type" << std::endl;
        return 1;
    }

    int gamma_num = 100; // energy numbers
    std::vector<long double> gmes = Logspace(1, 9, gamma_num);


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