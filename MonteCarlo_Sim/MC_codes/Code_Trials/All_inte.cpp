#include <iostream>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <yaml-cpp/yaml.h>
//#include <time.h>
//#include <omp.h>

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
        betaD = (Gamma1 * Gamma1) * Dbeta *(1 + Gamma1 * Gamma1 * Dbeta * beta1);
        gme2 = g_me * (1 + 0.5 * (betaD * betaD) - betae * betaD * costheta);
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
long double tau (long double gme, long double B0, long double q, long double Lam_max, long double xi)
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

int main()
{   
    auto start_t = std::chrono::steady_clock::now();
    // load path
    std::string configPath = "/home/wsy/Acc_MC/MC_sim/paras.yaml";
    std::string outputDir = "/home/wsy/Acc_MC/MC_sim/codes/Code_Trials/cpp_results/";
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

    int Num_steps = 200; // angle numbers
    int gamma_num = 21; // energy numbers

    std::vector<long double>Gamma_list(gamma_num);
    std::vector<long double>Costheta_list(Num_steps);
    std::vector<long double>Alpha_list(Num_steps);
    std::vector<long double>Phi_list(Num_steps);
    long double res[Num_steps * Num_steps * Num_steps][gamma_num] = {0};

    Gamma_list = Logspace(1, 9, gamma_num);
    Costheta_list = Linspace(-1, 1, Num_steps);
    Alpha_list = Linspace(0,2*pi, Num_steps);
    Phi_list = Linspace(0, 2*pi, Num_steps);

    for (int i = 0; i < gamma_num; ++i) // loop for energy

    {   int counts = 0;
        long double gme0 = Gamma_list[i];
        long double dt = tau(gme0, B0 , q, Lam_max, xi);
        //printf("%.2f\n", dt);

        for (int j = 0; j < Num_steps; ++j) // loop for costheta
        {
            long double costheta = Costheta_list[j];

            for (int k =0; k < Num_steps; ++k) // loop for alpha
            {
                long double alpha = Alpha_list[k];

                for (int l = 0; l < Num_steps; ++l) // loop for phi
                {   
                    long double phi = Phi_list[l];
                    long double g_me;
                    long double beta_ini = Beta_Dis (r0, R_sh, GM0, eta, beta_min);

                    //printf("%.2f\n", phi);

                    if (Integrat == true) // direct integration
                    {
                        long double C_A = Coeff_A ( beta_ini, GM0, eta, R_sh);
                        long double tau = (gme0, B0, q, Lam_max, xi);
                        long double sintheta = std::sqrt(1 - costheta * costheta);
                        long double betae = std::sqrt(1 - 1/ pow(gme0, 2));
                        g_me = gme0 * (1 + 0.5 * pow (C_A , 2) * pow (betae, 2) * pow(tau, 2) * pow (sintheta, 2) * pow (std::cos(alpha), 2) 
                            - C_A * pow (betae, 2) * tau * sintheta * costheta * std::cos(alpha)
                            - pow(C_A, 2) * beta_ini * pow (betae , 3) * tau * pow (sintheta , 2) * costheta * pow(std::cos(alpha),2));
                    }
                    else
                    {

                    
                        long double r_tmp;
                        long double beta_tmp;

                        if (Move_type == 1)
                        {   
                            long double x = r0 * std::cos(alpha); long double y = r0 * std::sin(alpha);
                            //long double r_ini = std::sqrt(x * x + y * y);
                            auto [renew_x, renew_y] = movement_e(gme0 , costheta , phi, x , y, dt);
                            r_tmp = std::sqrt(renew_x * renew_x + renew_y * renew_y);
                            beta_tmp = Beta_Dis(r_tmp, R_sh, GM0, eta, beta_min);
                            //printf("%.20f\n", beta_ini);
                        }
                        else if (Move_type == 2) // no approximation
                        {
                            double dr = movement_e2 (gme0, costheta, alpha, phi, r0, dt, false);
                            r_tmp = r0 + dr;
                            beta_tmp = Beta_Dis(r_tmp, R_sh, GM0, eta, beta_min);
                        }
                        else if (Move_type == 3) // with approximation
                        {   
                            //long double sintheta = std::sqrt (1 - costheta * costheta);
                            //long double dx = std::sqrt(1 - 1/(gme0 * gme0)) * sintheta * std::cos(phi) *c *dt;
                            double dr = movement_e2 (gme0, costheta, alpha, phi, r0, dt, true);
                            r_tmp = r0 + dr;
                            beta_tmp = Beta_Dis(r_tmp, R_sh, GM0, eta, beta_min);

                        }
                        g_me = LorentzGamma (beta_ini, beta_tmp, costheta, gme0, E_approx);
                    }

                    res[counts][i] = g_me;
                    //printf("%.15Lf\n", g_me);
                    counts += 1;
                    
                }
            }

        }
    printf("%s\n", "Done !");
    }

    // save to a text file
    std::string outputFile = outputDir + "results.txt";
    std::ofstream outFile(outputFile);

    outFile << std::fixed << std::setprecision(30); //high precision

    if (!outFile.is_open())
    {
        std::cerr << "Error occurred while opening file :" << outputFile << std::endl;
        return 1;
    }

    for (int i = 0; i < Num_steps * Num_steps * Num_steps; ++i)
    {
        for (int j =0; j < gamma_num; ++j)
        {
            outFile << res[i][j] << (j < gamma_num - 1 ? " " : "\n"); // change rows

        }
    }

    outFile.close();
    std::cout << "Results saved !" << std::endl;

    auto end_t = std::chrono::steady_clock::now();
    double duration = std::chrono::duration<double>(end_t - start_t).count();
    std::cout << "Running time:" << duration << "s" << std::endl;
    return 0;

}