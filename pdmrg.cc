#include "parallel_dmrg.h"
#include "itensor/all.h"

using namespace itensor;

int
main(int argc, char* argv[])
    {
    Environment env(argc,argv);

    parallelDebugWait(env);

    int N = 100;

    SpinOne sites;
    IQMPO H;
    IQMPS psi;
    Sweeps sweeps;
    if(env.firstNode())
        {
        sites = SpinOne(N); //make a chain of N spin 1's
        auto ampo = AutoMPO(sites);
        for(int j = 1; j < N; ++j)
            {
            ampo += 0.5,"S+",j,"S-",j+1;
            ampo += 0.5,"S-",j,"S+",j+1;
            ampo +=     "Sz",j,"Sz",j+1;
            }
        H = IQMPO(ampo);
        auto state = InitState(sites);
        for(auto n : range1(N)) state.set(n,n%2==1?"Up":"Dn");
        psi = IQMPS(state);
        psi.normalize();

        sweeps = Sweeps(5);
        sweeps.maxm() = 10,20,100,100,200;
        sweeps.cutoff() = 1E-10;
        sweeps.niter() = 2;
        println(sweeps);
        }
    env.broadcast(sites,H,psi,sweeps);

    parallel_dmrg(env,psi,H,sweeps,"Quiet");

    return 0;
    }
