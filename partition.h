//
// Distributed under the ITensor Library License, Version 1.1.
//    (See accompanying LICENSE file.)
//
#ifndef __ITENSOR_PARTITION_H
#define __ITENSOR_PARTITION_H

//#include "itensor/global.h"
//#include "itensor/util/print_macro.h"
#include "itensor/all.h"
//#include "itensor/util/readwrite.h"

namespace itensor {

//
// Partition
//
// Represents a division of N sites
// into Nb blocks.
//
// Blocks are numbered 1,2,..,Nb.
//
//

class Partition
    {
    int N_ = 0; 
    int Nb_ = 0;
    std::vector<int> bound_;
    public:

    Partition();

    Partition(int N, int Nb);

    Partition(int N, int Nb, int bound_size);

    int
    Nb() const { return Nb_; }

    int
    begin(int block) const;
    int
    end(int block) const;
    int
    size(int block) const;

    bool
    hasSite(int block, int s) const;
    bool
    hasBond(int block, int b) const;

    void
    read(std::istream & s);
    void
    write(std::ostream & s) const;

    };

inline Partition::
Partition()
    :
    N_(0),
    Nb_(0)
    { }

inline Partition::
Partition(int N, int Nb)
    :
    N_(N),
    Nb_(Nb),
    bound_(Nb_)
    {
    //Enlarge first block if N_ not a
    //multiple of Nb_
    int step = N_/Nb_,
        extra = N_%Nb_/2;
    bound_.at(0) = step+extra;
    for(int cut_ = bound_.at(0), j = 1; j < Nb_-1; ++j)
        {
        cut_ += step;
        bound_.at(j) = cut_;
        }
    }

inline Partition::
Partition(int N, int Nb, int bound_size)
    :
    N_(N),
    Nb_(Nb),
    bound_(Nb_)
    {
    if(Nb <= 2 && (bound_size * Nb) != N)
        {
        Error("Cannot honor bound_size request");
        }
    //Set requested boundary size
    bound_.at(0) = bound_size;

    //Use same algorithm as above but with
    //the user-requested boundaries subtracted out
    //(and 2 fewer blocks)
    int Neff = N-2*bound_size,
        step = Neff/(Nb_-2),
        extra = Neff%(Nb_-2)/2;
    if(step < 1 || Neff < 1)
        {
        Print(Neff);
        Print(step);
        Print(extra);
        Error("Ill-formed Partition");
        }
    bound_.at(1) = bound_size + step + extra;
    for(int cut_ = bound_.at(1), j = 2; j < Nb_-1; ++j)
        {
        cut_ += step;
        if(j == Nb_-2) cut_ += extra;
        bound_.at(j) = cut_;
        }
    }


int inline Partition::
begin(int block) const
    {
    if(block == 1) return 1;
    return bound_.at(block-2)+1;
    }

int inline Partition::
end(int block) const
    {
    if(block == Nb_) return N_;
    return bound_.at(block-1);
    }

int inline Partition::
size(int block) const
    {
    return (end(block)-begin(block)+1);
    }

bool inline Partition::
hasSite(int block, int s) const
    {
    return (s >= begin(block) && s <= end(block));
    }
bool inline Partition::
hasBond(int block, int b) const
    {
    return (b >= begin(block) && b < end(block));
    }

void inline Partition::
read(std::istream & s)
    {
    itensor::read(s,N_);
    itensor::read(s,Nb_);
    itensor::read(s,bound_);
    }

void inline Partition::
write(std::ostream & s) const
    {
    itensor::write(s,N_);
    itensor::write(s,Nb_);
    itensor::write(s,bound_);
    }

inline 
std::ostream& 
operator<<(std::ostream& s, const Partition& p)
    {
    s << "Partition:\n";
    if(p.Nb() == 0) s << " (empty)\n";
    for(int b = 1; b <= p.Nb(); ++b)
        {
        s << format("%d (%d,%d) %d\n",b,p.begin(b),p.end(b),p.size(b));
        }
    s << std::endl;
    return s;
    }

} //namespace itensor

#endif
