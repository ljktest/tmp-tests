
#ifndef HYSOP_INDEX_SEQ_H
#define HYSOP_INDEX_SEQ_H

namespace hysop {
    namespace detail {

        template <int...>
            struct index_seq {};
        
        template <int k, std::size_t d, int... I>
            struct constant_seq_impl {
                typedef typename constant_seq_impl<k,d-1,k,I...>::type type;
            };
        
        template <int k, int... I>
            struct constant_seq_impl<k,0,I...> {
                typedef index_seq<I...> type;
            };

        template <std::size_t count, int step, int current, int... I>
            struct index_seq_impl {
                typedef typename index_seq_impl<count-1,step,current+step,I...,current>::type type;
            };
        template <int step, int current, int... I>
            struct index_seq_impl<0,step,current,I...> {
                typedef index_seq<I...> type;
            };
        
        
        template <std::size_t count, int i0=0, int step=1>
            using index_seq_gen = typename index_seq_impl<count,step,i0>::type;
        
        template <int constant, std::size_t count>
            using constant_seq_gen = typename constant_seq_impl<constant,count>::type;
    }
}



#endif /* end of include guard: HYSOP_INDEX_SEQ_H */

