#ifndef __ITENSOR_PARALLEL_DMRG
#define __ITENSOR_PARALLEL_DMRG

#include "itensor/mps/dmrg.h"
#include "itensor/util/parallel.h"
#include "partition.h"

namespace itensor {

template<typename Tensor>
Spectrum
isvd(Tensor T, 
     Tensor & A, 
     Tensor & V, 
     Tensor & B,
     Args const& args = Args::global())
    {
    auto pinv_cut = args.getReal("PInvCut",1E-12);
    auto spec = svd(T,A,V,B,args);
    A *= V;
    B *= V;
    auto pseudoInv = [pinv_cut](Real x)
        {
        return (std::fabs(x) <= pinv_cut) ? 0.0 : 1./x;
        };
    V.apply(pseudoInv);
    V.dag();
    return spec;
    }

struct ParallelSweeper
    {
    int j = -1;
    int jl = -1;
    int jr = -1;
    int node = -1; //1-indexed
    int ha = 1; //records 1st or 2nd half sweep
    int sw = 0; //number of full sweeps done

    ParallelSweeper(int jl_,
                    int jr_,
                    int node_)
      : jl(jl_),
        jr(jr_),
        node(node_)
        { 
        if(node <= 0) Error("Node number out of range");
        j = odd() ? jl : (jr-1);
        }

    void
    operator++()
        {
        int inc = 0;
        int end = 0;
        if(odd())
            {
            inc = (ha==1 ? +1 : -1);
            end = (ha==1 ? jr : jl-1);
            }
        else
            {
            inc = (ha==1 ? -1 : +1);
            end = (ha==1 ? jl-1 : jr);
            }
        j += inc;
        if(j == end)
            {
            j -= inc; //went 1 too far, fix
            ++ha;
            }
        }

    void
    newSweep() { ha = 1; }
    bool
    doingHalf() const { return ha == 1; }
    bool
    doingFull() const { return ha < 3; }

    Direction
    dir() const
        {
        if(odd()) return (ha==1 ? Fromleft : Fromright);
        return (ha==1 ? Fromright : Fromleft);
        }

    bool
    atRight() const { return j == jr-1; }
    bool
    atLeft() const { return j == jl; }

    private:

    bool
    odd() const { return node%2==1; }

    };

template <typename Tensor, typename HamT>
Real 
pdmrgWorker(Environment const& env,
            Partition const& P,
            MPSt<Tensor> & psi,
            HamT & PH,
            Sweeps const& sweeps,
            Args args = Args::global());

template <typename Tensor>
LocalMPO<Tensor> 
computeHEnvironment(Environment const& env,
                    Partition const& P,
                    MPSt<Tensor> const& psi,
                    std::vector<Tensor> const& Vs,
                    MPOt<Tensor> const& H,
                    Args const& args = Args::global())
    {
    std::vector<Tensor> LH;
    std::vector<Tensor> RH;
    if(env.firstNode())
        {
        auto Nnode = env.nnodes();
        LH = std::vector<Tensor>(Nnode+1);
        RH = std::vector<Tensor>(Nnode+1);

        //Make left Hamiltonian environments
        auto L = Tensor(1.);
        LH.at(1) = L;
        for(int b = 1; b < P.Nb(); ++b)
            {
            for(auto j : range1(P.begin(b),P.end(b)))
                {
                L = L*psi.A(j)*H.A(j)*dag(prime(psi.A(j)));
                }
            L *= Vs.at(b);
            L *= dag(prime(Vs.at(b)));
            LH.at(b+1) = L;
            //printfln("LH[%d] = \n%s",b+1,LH[b+1]);
            //printfln("psi.A(%d) = \n%s",P.end(b)+1,psi.A(P.end(b)+1));
            }
        //Make right Hamiltonian environments
        auto R = Tensor(1.);
        RH.at(P.Nb()) = R;
        for(int b = P.Nb(); b > 1; --b)
            {
            for(auto j = P.end(b); j >= P.begin(b); --j)
                {
                R = R*psi.A(j)*H.A(j)*dag(prime(psi.A(j)));
                }
            R *= Vs.at(b-1);
            R *= dag(prime(Vs.at(b-1)));
            RH.at(b-1) = R;
            //printfln("RH[%d] = \n%s",b-1,RH[b-1]);
            }
        }
    env.broadcast(LH,RH);

    auto b = env.rank()+1; //block number of this node
    return LocalMPO<Tensor>(H,LH.at(b),P.begin(b)-1,RH.at(b),P.end(b)+1,args);
    }

template <typename Tensor>
LocalMPOSet<Tensor> 
computeHEnvironment(Environment const& env,
                    Partition const& P,
                    MPSt<Tensor> const& psi,
                    std::vector<Tensor> const& Vs,
                    std::vector<MPOt<Tensor>> const& Hset,
                    Args const& args = Args::global())
    {
    auto nset = Hset.size();
    auto Nnode = env.nnodes();
    std::vector<std::vector<Tensor>> LH(Nnode+1);
    std::vector<std::vector<Tensor>> RH(Nnode+1);
    if(env.firstNode())
        {
        for(auto& lh : LH) lh = std::vector<Tensor>(nset);
        for(auto& rh : RH) rh = std::vector<Tensor>(nset);

        for(auto n : range(nset))
            {
            auto& H = Hset.at(n);

            //Make left Hamiltonian environments
            auto L = Tensor(1.);
            LH.at(1).at(n) = L;
            for(int b = 1; b < P.Nb(); ++b)
                {
                for(auto j : range1(P.begin(b),P.end(b)))
                    {
                    L = L*psi.A(j)*H.A(j)*dag(prime(psi.A(j)));
                    }
                L *= Vs.at(b);
                L *= dag(prime(Vs.at(b)));
                LH.at(b+1).at(n) = L;
                //printfln("LHn[%d] = \n%s",b+1,LHn[b+1]);
                //printfln("psi.A(%d) = \n%s",P.end(b)+1,psi.A(P.end(b)+1));
                }
            //Make right Hamiltonian environments
            auto R = Tensor(1.);
            RH.at(P.Nb()).at(n) = R;
            for(int b = P.Nb(); b > 1; --b)
                {
                for(auto j = P.end(b); j >= P.begin(b); --j)
                    {
                    R = R*psi.A(j)*H.A(j)*dag(prime(psi.A(j)));
                    }
                R *= Vs.at(b-1);
                R *= dag(prime(Vs.at(b-1)));
                RH.at(b-1).at(n) = R;
                //printfln("RHn[%d] = \n%s",b-1,RHn[b-1]);
                }
            }
        }
    for(auto r : range(Nnode+1))
        {
        env.broadcast(LH.at(r),RH.at(r));
        }

    auto b = env.rank()+1; //block number of this node
    return LocalMPOSet<Tensor>(Hset,LH.at(b),P.begin(b)-1,RH.at(b),P.end(b)+1,args);
    }

template<typename Tensor>
void
splitWavefunction(Environment const& env,
                  MPSt<Tensor> & psi, 
                  Partition & P,
                  std::vector<Tensor> & Vs)
    {
    if(env.firstNode()) 
        {
        auto Nnode = env.nnodes();
        P = Partition(psi.N(),Nnode);
        println(P);

        Vs = std::vector<Tensor>(Nnode);
        psi.position(1);
        auto c = 1;
        for(int b = 1; b < P.Nb(); ++b)
            {
            auto n = P.end(b);
            //Shift ortho center to one past the end of the b'th block
            while(c < n+1)
                {
                Tensor D;
                svd(psi.A(c)*psi.A(c+1),psi.Aref(c),D,psi.Aref(c+1));
                psi.Aref(c+1) *= D;
                c += 1;
                }
            if(c != n+1) Error("c != n+1");
            auto AA = psi.A(n)*psi.A(n+1);
            auto& V = Vs.at(b);
            isvd(AA,psi.Aref(n),V,psi.Aref(n+1));
            }
        }
    env.broadcast(P,Vs,psi);
    }


//
// parallel_dmrg with single MPO or IQMPO
//
template <typename Tensor>
Real 
parallel_dmrg(Environment const& env,
              MPSt<Tensor> & psi,
              MPOt<Tensor> const& H,
              Sweeps const& sweeps,
              Args args = Args::global())
    {
    Partition P;
    std::vector<Tensor> Vs;
    splitWavefunction(env,psi,P,Vs);
    auto PH = computeHEnvironment(env,P,psi,Vs,H,args);
    return pdmrgWorker(env,P,psi,Vs,PH,sweeps,args);
    }

//
// parallel_dmrg with an (implicit) sum of MPOs or IQMPOs
//
template <typename Tensor>
Real 
parallel_dmrg(Environment const& env,
              MPSt<Tensor> & psi,
              std::vector<MPOt<Tensor>> const& Hset,
              Sweeps const& sweeps,
              Args args = Args::global())
    {
    Partition P;
    std::vector<Tensor> Vs;
    splitWavefunction(env,psi,P,Vs);
    auto PH = computeHEnvironment(env,P,psi,Vs,Hset,args);
    return pdmrgWorker(env,P,psi,Vs,PH,sweeps,args);
    }

template<typename Tensor, typename HType = Tensor>
struct Boundary
    {
    HType HH;
    Tensor A;
    Tensor UU;
    Real energy;

    Boundary() : energy(0) { }

    void
    write(std::ostream& s) const
        {
        itensor::write(s,HH);
        itensor::write(s,A);
        itensor::write(s,UU);
        itensor::write(s,energy);
        }
    void
    read(std::istream& s)
        {
        itensor::read(s,HH);
        itensor::read(s,A);
        itensor::read(s,UU);
        itensor::read(s,energy);
        }
    };

template <typename Tensor, typename HamT>
Real 
pdmrgWorker(Environment const& env,
            Partition const& P,
            MPSt<Tensor> & psi,
            std::vector<Tensor> & Vs,
            HamT & PH,
            Sweeps const& sweeps,
            Args args)
    {
    using EdgeType = stdx::decay_t<decltype(PH.L())>;
    //Maximum number of Davidson iterations at boundaries
    auto boundary_niter = args.getInt("BoundaryIter",sweeps.niter(1)+6);

    auto b = env.rank()+1;
    auto jl = P.begin(b);
    auto jr = P.end(b);

    Real energy = 0.;

    auto psw = ParallelSweeper(jl,jr,b);

    MailBox mboxL,mboxR;
    if(not env.firstNode()) mboxL = MailBox(env,env.rank()-1);
    if(not env.lastNode())  mboxR = MailBox(env,env.rank()+1);

    psi.leftLim(jl);
    psi.rightLim(jr);
    psi.position(psw.j);

    for(int sw = 1; sw <= sweeps.nsweep(); ++sw)
        {
        args.add("Sweep",sw);
        args.add("Cutoff",sweeps.cutoff(sw));
        args.add("Minm",sweeps.minm(sw));
        args.add("Maxm",sweeps.maxm(sw));
        args.add("Noise",sweeps.noise(sw));
        args.add("MaxIter",sweeps.niter(sw));

        printfln("Doing sweep %d for node %d",sw,b);

        for(psw.newSweep(); psw.doingFull(); ++psw)
            {
            auto j = psw.j;
            auto dir = psw.dir();
            //printfln("%d j = %d (%d,%d) %s",b,j,jl,jr,dir==Fromleft?"Fromleft":"Fromright");
            PH.position(j,psi);

            auto phi = psi.A(j)*psi.A(j+1);

            energy = davidson(PH,phi,args);

            //if(env.rank()+1 == 1) printfln("%s j = %d energy = %.10f",dir==Fromleft?"->":"<-",j,energy);
            
            auto spec = psi.svdBond(j,phi,dir,PH,args);

            if(psw.atRight() && dir==Fromleft && bool(mboxR))
                {
                auto prev_energy = energy;
                printfln("Node %d communicating with right, boundary_niter=%d",b,boundary_niter);
				auto n = j+1;

                PH.position(n,psi);

                Boundary<Tensor,EdgeType> B;
                mboxR.receive(B);
                psi.Aref(n+1) = B.A;
                PH.R(B.HH);
                B = Boundary<Tensor,EdgeType>(); //to save memory

                auto& V = Vs.at(b);

                auto phi = psi.A(n)*V*psi.A(n+1);
                phi /= norm(phi);

                energy = davidson(PH,phi,{args,"MaxIter=",boundary_niter});

                auto spec = isvd(phi,psi.Aref(n),V,psi.Aref(n+1));

                B.HH = PH.L();
                B.UU = psi.A(n)*V;
                B.A = psi.A(n+1);
                B.energy = energy;
                mboxR.send(B);

                psi.Aref(n+1) *= V;
                psi.rightLim(n+1);

                printfln("Node %d done with boundary step, energy %.5f -> %.5f",b,prev_energy,energy);
                }
            else if(psw.atLeft() && dir==Fromright && bool(mboxL))
                {
                auto prev_energy = energy;
                printfln("Node %d communicating with left, boundary_niter=%d",b,boundary_niter);

                auto n = j-1;

                PH.position(n,psi);

                Boundary<Tensor,EdgeType> B;
                B.A = psi.A(n+1);
                B.HH = PH.R();
                mboxL.send(B);

                mboxL.receive(B);
                PH.L(B.HH);
                PH.shift(n,Fromleft,B.UU);
                psi.Aref(n+1) = B.A;
                energy = B.energy;
                printfln("Node %d done with boundary step, energy %.5f -> %.5f",b,prev_energy,energy);
                }
            } //for loop over b
        }

    printfln("Block %d final energy = %.12f",b,energy);

    return energy;

    } // pdmrgWorker

} //namespace itensor

#endif
