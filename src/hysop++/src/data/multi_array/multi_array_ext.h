
#ifndef HYSOP_MULTI_ARRAY_H
#include "data/multi_array/multi_array.h"
#else 

#ifndef HYSOP_MULTI_ARRAY_EXT_H
#define HYSOP_MULTI_ARRAY_EXT_H

namespace hysop {
    namespace data {

        /* distances */
        template <typename T, std::size_t Dim> 
            T distance_L1(const const_multi_array_ref<T,Dim> &lhs, const const_multi_array_ref<T,Dim> &rhs);
        template <typename T, std::size_t Dim>
            T distance_L2(const const_multi_array_ref<T,Dim> &lhs, const const_multi_array_ref<T,Dim> &rhs);
        template <typename T, std::size_t Dim>
            T distance_Linf(const const_multi_array_ref<T,Dim> &lhs, const const_multi_array_ref<T,Dim> &rhs);

        /* unary operators */
        template <typename T, std::size_t Dim>
        multi_array<T,Dim> operator+(const const_multi_array_ref<T,Dim>& arr);
        template <typename T, std::size_t Dim>
        multi_array<T,Dim> operator-(const const_multi_array_ref<T,Dim>& arr);
        template <std::size_t Dim>
        multi_array<bool,Dim> operator~(const const_multi_array_ref<bool,Dim>& arr);

        /* elementwise arithmetic operations */
        template <typename T, std::size_t Dim>
            multi_array<T,Dim> operator-(const const_multi_array_ref<T,Dim>& lhs, const const_multi_array_ref<T,Dim>& rhs);
        template <typename T, std::size_t Dim>
            multi_array<T,Dim> operator+(const const_multi_array_ref<T,Dim>& lhs, const const_multi_array_ref<T,Dim>& rhs);
        template <typename T, std::size_t Dim>
            multi_array<T,Dim> operator*(const const_multi_array_ref<T,Dim>& lhs, const const_multi_array_ref<T,Dim>& rhs);
        template <typename T, std::size_t Dim>
            multi_array<T,Dim> operator/(const const_multi_array_ref<T,Dim>& lhs, const const_multi_array_ref<T,Dim>& rhs);
        template <typename T, std::size_t Dim>
            multi_array<T,Dim> operator%(const const_multi_array_ref<T,Dim>& lhs, const const_multi_array_ref<T,Dim>& rhs);

        /* elementwise boolean operations */
        template <typename T, std::size_t Dim>
            multi_array<bool,Dim> operator& (const const_multi_array_ref<T,Dim>& lhs, const const_multi_array_ref<T,Dim>& rhs);
        template <typename T, std::size_t Dim>
            multi_array<bool,Dim> operator| (const const_multi_array_ref<T,Dim>& lhs, const const_multi_array_ref<T,Dim>& rhs);
        template <typename T, std::size_t Dim>
            multi_array<bool,Dim> operator^ (const const_multi_array_ref<T,Dim>& lhs, const const_multi_array_ref<T,Dim>& rhs);

        
        /* element wise ordering test - near equality test for floating point types, see utils/utils.h::areEqual<T> */
        template <typename T, std::size_t Dim>
            multi_array<bool,Dim> operator==(const const_multi_array_ref<T,Dim>& lhs, const const_multi_array_ref<T,Dim>& rhs);
        template <typename T, std::size_t Dim>
            multi_array<bool,Dim> operator!=(const const_multi_array_ref<T,Dim>& lhs, const const_multi_array_ref<T,Dim>& rhs);
        template <typename T, std::size_t Dim>
            multi_array<bool,Dim> operator>=(const const_multi_array_ref<T,Dim>& lhs, const const_multi_array_ref<T,Dim>& rhs);
        template <typename T, std::size_t Dim>
            multi_array<bool,Dim> operator<=(const const_multi_array_ref<T,Dim>& lhs, const const_multi_array_ref<T,Dim>& rhs);
        template <typename T, std::size_t Dim>
            multi_array<bool,Dim> operator> (const const_multi_array_ref<T,Dim>& lhs, const const_multi_array_ref<T,Dim>& rhs);
        template <typename T, std::size_t Dim>
            multi_array<bool,Dim> operator< (const const_multi_array_ref<T,Dim>& lhs, const const_multi_array_ref<T,Dim>& rhs);



        /* And all their view and rvalue references variants */
        /* Ref    - Ref                       */
        /* Ref    - View                      */
        /* View   - Ref                       */
        /* View   - View                      */
        /*                                    */
        /* Rvalue - View                      */
        /* Rvalue - Ref                       */
        /* View   - Rvalue                    */
        /* Ref    - Rvalue                    */
        /* ...                                */

        
        /* Implementation */
        
        #define CHECK_SHAPES(LHS,RHS) assert(LHS.shape() == RHS.shape())

        #define BINARY_OP(TEMPLATES,T,R,RES,OPNAME,BINOP,LHS,RHS,LOOP_OVER_ALL_ELEMENTS,DATA_ACCESS,CREATE_BUFFER,FROM_BUFFER,RET_OP) \
            TEMPLATES                                                                                                             \
                RES OPNAME(LHS lhs, RHS rhs) {                                                                                    \
                    CHECK_SHAPES(lhs,rhs);                                                                                        \
                    CREATE_BUFFER(FROM_BUFFER,R);                                                                                 \
                    LOOP_OVER_ALL_ELEMENTS(lhs) {                                                                                 \
                        const T& lhsVal = DATA_ACCESS(lhs);                                                                       \
                        const T& rhsVal = DATA_ACCESS(rhs);                                                                       \
                        BINOP;                                                                                                    \
                    }                                                                                                             \
                    return RET_OP(BUFFER_NAME);                                                                                   \
                }

        #define BINARY_OP_VIEW_VIEW(TEMPLATES,T,R,RET,OPNAME,BINOP,CREATE_BUFFER,RET_OP)   \
                BINARY_OP(SINGLE_ARG(TEMPLATES),SINGLE_ARG(T),SINGLE_ARG(R),SINGLE_ARG(RET),OPNAME,SINGLE_ARG(BINOP),\
                        VIEW(T),VIEW(T),LOOP_OVER_ALL_VIEW_ELEMENTS,VIEW_DATA_ACCESS,CREATE_BUFFER,lhs,RET_OP)
        #define BINARY_OP_VIEW_REF(TEMPLATES,T,R,RET,OPNAME,BINOP,CREATE_BUFFER,RET_OP)   \
                BINARY_OP(SINGLE_ARG(TEMPLATES),SINGLE_ARG(T),SINGLE_ARG(R),SINGLE_ARG(RET),OPNAME,SINGLE_ARG(BINOP),\
                        VIEW(T),REF(T),LOOP_OVER_ALL_VIEW_ELEMENTS,VIEW_DATA_ACCESS,CREATE_BUFFER,lhs,RET_OP)
        #define BINARY_OP_REF_VIEW(TEMPLATES,T,R,RET,OPNAME,BINOP,CREATE_BUFFER,RET_OP)   \
                BINARY_OP(SINGLE_ARG(TEMPLATES),SINGLE_ARG(T),SINGLE_ARG(R),SINGLE_ARG(RET),OPNAME,SINGLE_ARG(BINOP),\
                        REF(T),VIEW(T),LOOP_OVER_ALL_VIEW_ELEMENTS,VIEW_DATA_ACCESS,CREATE_BUFFER,lhs,RET_OP)
        #define BINARY_OP_REF_REF(TEMPLATES,T,R,RET,OPNAME,BINOP,CREATE_BUFFER,RET_OP)     \
                BINARY_OP(SINGLE_ARG(TEMPLATES),SINGLE_ARG(T),SINGLE_ARG(R),SINGLE_ARG(RET),OPNAME,SINGLE_ARG(BINOP),\
                        REF(T),REF(T),LOOP_OVER_ALL_REF_ELEMENTS,REF_DATA_ACCESS,CREATE_BUFFER,lhs,RET_OP)
        #define BINARY_OP_REF_RVALUE(TEMPLATES,T,R,RET,OPNAME,BINOP,CREATE_BUFFER,RET_OP)  \
                BINARY_OP(SINGLE_ARG(TEMPLATES),SINGLE_ARG(T),SINGLE_ARG(R),SINGLE_ARG(RET),OPNAME,SINGLE_ARG(BINOP),\
                        REF(T),RVALUE(T),LOOP_OVER_ALL_REF_ELEMENTS,REF_DATA_ACCESS,CREATE_BUFFER,rhs,RET_OP)
        #define BINARY_OP_RVALUE_REF(TEMPLATES,T,R,RET,OPNAME,BINOP,CREATE_BUFFER,RET_OP)  \
                BINARY_OP(SINGLE_ARG(TEMPLATES),SINGLE_ARG(T),SINGLE_ARG(R),SINGLE_ARG(RET),OPNAME,SINGLE_ARG(BINOP),\
                        RVALUE(T),REF(T),LOOP_OVER_ALL_REF_ELEMENTS,REF_DATA_ACCESS,CREATE_BUFFER,lhs,RET_OP)
        #define BINARY_OP_VIEW_RVALUE(TEMPLATES,T,R,RET,OPNAME,BINOP,CREATE_BUFFER,RET_OP) \
                BINARY_OP(SINGLE_ARG(TEMPLATES),SINGLE_ARG(T),SINGLE_ARG(R),SINGLE_ARG(RET),OPNAME,SINGLE_ARG(BINOP),\
                        VIEW(T),RVALUE(T),LOOP_OVER_ALL_VIEW_ELEMENTS,VIEW_DATA_ACCESS,CREATE_BUFFER,rhs,RET_OP)
        #define BINARY_OP_RVALUE_VIEW(TEMPLATES,T,R,RET,OPNAME,BINOP,CREATE_BUFFER,RET_OP) \
                BINARY_OP(SINGLE_ARG(TEMPLATES),SINGLE_ARG(T),SINGLE_ARG(R),SINGLE_ARG(RET),OPNAME,SINGLE_ARG(BINOP),\
                        RVALUE(T),VIEW(T),LOOP_OVER_ALL_VIEW_ELEMENTS,VIEW_DATA_ACCESS,CREATE_BUFFER,lhs,RET_OP)

        #define LVALUE_BINARY_OPS(TEMPLATES,T,R,RET,OPNAME,BINOP,CREATE_BUFFER,RET_OP)                                         \
                BINARY_OP_VIEW_VIEW(SINGLE_ARG(TEMPLATES),SINGLE_ARG(T),SINGLE_ARG(R),SINGLE_ARG(RET),OPNAME,SINGLE_ARG(BINOP),CREATE_BUFFER,RET_OP)   \
                BINARY_OP_VIEW_REF(SINGLE_ARG(TEMPLATES),SINGLE_ARG(T),SINGLE_ARG(R),SINGLE_ARG(RET),OPNAME,SINGLE_ARG(BINOP),CREATE_BUFFER,RET_OP)    \
                BINARY_OP_REF_VIEW(SINGLE_ARG(TEMPLATES),SINGLE_ARG(T),SINGLE_ARG(R),SINGLE_ARG(RET),OPNAME,SINGLE_ARG(BINOP),CREATE_BUFFER,RET_OP)    \
                BINARY_OP_REF_REF(SINGLE_ARG(TEMPLATES),SINGLE_ARG(T),SINGLE_ARG(R),SINGLE_ARG(RET),OPNAME,SINGLE_ARG(BINOP),CREATE_BUFFER,RET_OP)

        #define RVALUE_BINARY_OPS(TEMPLATES,T,R,RET,OPNAME,BINOP,CREATE_BUFFER,RET_OP)                                         \
                BINARY_OP_REF_RVALUE(SINGLE_ARG(TEMPLATES),SINGLE_ARG(T),SINGLE_ARG(R),SINGLE_ARG(RET),OPNAME,SINGLE_ARG(BINOP),CREATE_BUFFER,RET_OP)  \
                BINARY_OP_RVALUE_REF(SINGLE_ARG(TEMPLATES),SINGLE_ARG(T),SINGLE_ARG(R),SINGLE_ARG(RET),OPNAME,SINGLE_ARG(BINOP),CREATE_BUFFER,RET_OP)  \
                BINARY_OP_VIEW_RVALUE(SINGLE_ARG(TEMPLATES),SINGLE_ARG(T),SINGLE_ARG(R),SINGLE_ARG(RET),OPNAME,SINGLE_ARG(BINOP),CREATE_BUFFER,RET_OP) \
                BINARY_OP_RVALUE_VIEW(SINGLE_ARG(TEMPLATES),SINGLE_ARG(T),SINGLE_ARG(R),SINGLE_ARG(RET),OPNAME,SINGLE_ARG(BINOP),CREATE_BUFFER,RET_OP)
      
       
        /* Code generator */
        #define BUFFER_NAME macro_generated_local_buffer
        #define CREATE_MA_BUFFER(FROM_MA,R) multi_array<R,Dim>  BUFFER_NAME(FROM_MA.shape())
        #define CREATE_R_BUFFER(FROM_MA,R)  R BUFFER_NAME = R(0)
        #define NO_MA_BUFFER(FROM_MA,R)     multi_array<R,Dim>& BUFFER_NAME = FROM_MA

        #define SIMPLE_BINOP(OP)     DATA_ACCESS(BUFFER_NAME) = ((lhsVal) OP (rhsVal))
        #define SIMPLE_FBINOP(FOP)   DATA_ACCESS(BUFFER_NAME) = FOP((lhsVal),(rhsVal))
        #define SUP_OR_EQUAL()       DATA_ACCESS(BUFFER_NAME) = (((lhsVal) > (rhsVal)) || hysop::utils::areEqual<T>((rhsVal),(lhsVal)))
        #define INF_OR_EQUAL()       DATA_ACCESS(BUFFER_NAME) = (((lhsVal) < (rhsVal)) || hysop::utils::areEqual<T>((rhsVal),(lhsVal)))
        #define IDENTITY(X) (X)
        
        #define T1         SINGLE_ARG(template<std::size_t Dim>)
        #define T1bis      SINGLE_ARG(template<std::size_t Dim, typename Allocator>)
        #define T2         SINGLE_ARG(template<typename T, std::size_t Dim>)
        #define T3         SINGLE_ARG(template<typename T, std::size_t Dim, typename Allocator>)
        #define VIEW(T)    SINGLE_ARG(const const_multi_array_view<T,Dim>&)
        #define REF(T)     SINGLE_ARG(const const_multi_array_ref<T,Dim>&)
        #define RVALUE(T)  SINGLE_ARG(multi_array<T,Dim,Allocator>&&)
        #define LVALUE(T)  SINGLE_ARG(multi_array<T,Dim,Allocator>)
        #define DLVALUE(T) SINGLE_ARG(multi_array<T,Dim>)


        /* distances */
        LVALUE_BINARY_OPS(T2,T,T,T,distance_L1,\
                const T val = std::abs<T>(rhsVal-lhsVal); BUFFER_NAME += val,\
                CREATE_R_BUFFER,IDENTITY)
        LVALUE_BINARY_OPS(T2,T,T,T,distance_L2,\
                const T val = std::abs<T>(rhsVal-lhsVal); BUFFER_NAME += val*val,\
                CREATE_R_BUFFER,std::sqrt)
        LVALUE_BINARY_OPS(T2,T,T,T,distance_Linf,\
                SINGLE_ARG(const T val = std::abs<T>(rhsVal-lhsVal); BUFFER_NAME = std::max<T>(BUFFER_NAME,val)),\
                CREATE_R_BUFFER,IDENTITY)
        
        
        /* elementwise arithmetic operations */
        LVALUE_BINARY_OPS(T2,T,T,DLVALUE(T),operator+,SIMPLE_BINOP(+),CREATE_MA_BUFFER,IDENTITY)
        LVALUE_BINARY_OPS(T2,T,T,DLVALUE(T),operator-,SIMPLE_BINOP(-),CREATE_MA_BUFFER,IDENTITY)
        LVALUE_BINARY_OPS(T2,T,T,DLVALUE(T),operator*,SIMPLE_BINOP(*),CREATE_MA_BUFFER,IDENTITY)
        LVALUE_BINARY_OPS(T2,T,T,DLVALUE(T),operator/,SIMPLE_BINOP(/),CREATE_MA_BUFFER,IDENTITY)
        LVALUE_BINARY_OPS(T2,T,T,DLVALUE(T),operator%,SIMPLE_BINOP(%),CREATE_MA_BUFFER,IDENTITY)

        RVALUE_BINARY_OPS(T3,T,T,LVALUE(T),operator+,SIMPLE_BINOP(+),NO_MA_BUFFER,IDENTITY)
        RVALUE_BINARY_OPS(T3,T,T,LVALUE(T),operator-,SIMPLE_BINOP(-),NO_MA_BUFFER,IDENTITY)
        RVALUE_BINARY_OPS(T3,T,T,LVALUE(T),operator*,SIMPLE_BINOP(*),NO_MA_BUFFER,IDENTITY)
        RVALUE_BINARY_OPS(T3,T,T,LVALUE(T),operator/,SIMPLE_BINOP(/),NO_MA_BUFFER,IDENTITY)
        RVALUE_BINARY_OPS(T3,T,T,LVALUE(T),operator%,SIMPLE_BINOP(%),NO_MA_BUFFER,IDENTITY)
        
        /* elementwise boolean like operations */
        LVALUE_BINARY_OPS(T2,T,bool,DLVALUE(bool),operator&,SIMPLE_BINOP(&),CREATE_MA_BUFFER,IDENTITY)
        LVALUE_BINARY_OPS(T2,T,bool,DLVALUE(bool),operator|,SIMPLE_BINOP(|),CREATE_MA_BUFFER,IDENTITY)
        LVALUE_BINARY_OPS(T2,T,bool,DLVALUE(bool),operator^,SIMPLE_BINOP(^),CREATE_MA_BUFFER,IDENTITY)
        
        RVALUE_BINARY_OPS(T1bis,bool,bool,LVALUE(bool),operator&,SIMPLE_BINOP(&),NO_MA_BUFFER,IDENTITY)
        RVALUE_BINARY_OPS(T1bis,bool,bool,LVALUE(bool),operator|,SIMPLE_BINOP(|),NO_MA_BUFFER,IDENTITY)
        RVALUE_BINARY_OPS(T1bis,bool,bool,LVALUE(bool),operator^,SIMPLE_BINOP(^),NO_MA_BUFFER,IDENTITY)
        
        /* element wise ordering test - near equality test for floating point types, see utils/utils.h::areEqual<T> */
        LVALUE_BINARY_OPS(T2,T,bool,DLVALUE(bool),operator==,SIMPLE_FBINOP(hysop::utils::areEqual<T>),CREATE_MA_BUFFER,IDENTITY)
        LVALUE_BINARY_OPS(T2,T,bool,DLVALUE(bool),operator!=,SIMPLE_FBINOP(hysop::utils::areNotEqual<T>),CREATE_MA_BUFFER,IDENTITY)
        LVALUE_BINARY_OPS(T2,T,bool,DLVALUE(bool),operator<,SIMPLE_BINOP(<),CREATE_MA_BUFFER,IDENTITY)
        LVALUE_BINARY_OPS(T2,T,bool,DLVALUE(bool),operator>,SIMPLE_BINOP(>),CREATE_MA_BUFFER,IDENTITY)
        LVALUE_BINARY_OPS(T2,T,bool,DLVALUE(bool),operator>=,SUP_OR_EQUAL(),CREATE_MA_BUFFER,IDENTITY)
        LVALUE_BINARY_OPS(T2,T,bool,DLVALUE(bool),operator<=,INF_OR_EQUAL(),CREATE_MA_BUFFER,IDENTITY)
        
        /* Comparisson for booleans... */
        RVALUE_BINARY_OPS(T1bis,bool,bool,LVALUE(bool),operator==,SIMPLE_BINOP(==),NO_MA_BUFFER,IDENTITY)
        RVALUE_BINARY_OPS(T1bis,bool,bool,LVALUE(bool),operator!=,SIMPLE_BINOP(!=),NO_MA_BUFFER,IDENTITY)
        RVALUE_BINARY_OPS(T1bis,bool,bool,LVALUE(bool),operator<,SIMPLE_BINOP(<),NO_MA_BUFFER,IDENTITY)
        RVALUE_BINARY_OPS(T1bis,bool,bool,LVALUE(bool),operator>,SIMPLE_BINOP(>),NO_MA_BUFFER,IDENTITY)
        RVALUE_BINARY_OPS(T1bis,bool,bool,LVALUE(bool),operator>=,SIMPLE_BINOP(>=),NO_MA_BUFFER,IDENTITY)
        RVALUE_BINARY_OPS(T1bis,bool,bool,LVALUE(bool),operator<=,SIMPLE_BINOP(<=),NO_MA_BUFFER,IDENTITY)

        /* clean macros */
        #undef LVALUE_BINARY_OPS
        #undef RVALUE_BINARY_OPS

        #undef BINARY_OP_RVALUE_VIEW
        #undef BINARY_OP_RVALUE_REF
        #undef BINARY_OP_VIEW_RVALUE
        #undef BINARY_OP_REF_RVALUE

        #undef BINARY_OP_REF_REF
        #undef BINARY_OP_REF_VIEW
        #undef BINARY_OP_VIEW_REF
        #undef BINARY_OP_VIEW_VIEW

        #undef IDENTITY
        #undef SIMPLE_BINOP
        #undef SIMPLE_FBINOP
        #undef BINARY_OP

        #undef DLVALUE
        #undef LVALUE
        #undef RVALUE
        #undef REF
        #undef VIEW
        #undef T3
        #undef T2
        #undef T1bis
        #undef T1

        #undef NO_MA_BUFFER
        #undef CREATE_R_BUFFER
        #undef CREATE_MA_BUFFER
        #undef CHECK_SHAPES
        #undef BUFFER_NAME


    } /* end of namespace data */
} /* end of namespace hysop */

#endif /* end of include guard: HYSOP_MULTI_ARRAY_EXT_H */

#endif /* end of MULTI_ARRAY include guard */
