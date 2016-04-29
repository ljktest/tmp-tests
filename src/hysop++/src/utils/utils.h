
#ifndef HYSOP_UTILS_H
#define HYSOP_UTILS_H

#include <fftw3.h>
#include <array>
#include <vector>
#include <tuple>
#include <iostream>
#include <limits>

#include "maths/quad_maths.h"
#include "detail/index_seq.h"
#include "boost/multi_array.hpp"

namespace hysop {
    namespace utils {
       
        template <typename... T>
        void printTuple(std::ostream& os, const std::tuple<T...>& tuple);

        template <typename T>
            bool areEqual(const T &lhs, const T &rhs);
        template <typename T>
            bool areNotEqual(const T &lhs, const T &rhs);
       

        /* boost related utilities */
        template <std::size_t NumDims>
        boost::detail::multi_array::index_gen<NumDims, NumDims> buildView();

        template <std::size_t NumDims>
        boost::detail::multi_array::index_gen<NumDims, NumDims> buildIndices(
                const std::array<boost::multi_array_types::index_range, NumDims> &p_ranges);

        template <std::size_t NumRanges>
        boost::detail::multi_array::extent_gen<NumRanges> buildExtents(
                const std::array<std::size_t, NumRanges> &p_shape);


        /* Implementation */

        template <typename Tuple, int... I>
            void printTupleImpl(std::ostream& os, const Tuple& tuple, hysop::detail::index_seq<I...>) {
                const int dummy[sizeof...(I)] = { (os << std::get<I>(tuple) << ",", 0)... };
                os << std::get<sizeof...(I)>(tuple);
            }
        template <typename... T>
            void printTuple(std::ostream& os, const std::tuple<T...>& tuple) {
                os << "(";
                printTupleImpl(os,tuple, hysop::detail::index_seq_gen<sizeof...(T)-1>());
                os << ")";
            }
        
        template <typename T, typename>
            bool areEqualImpl(const T& lhs, const T& rhs) {
                return lhs == rhs;
            }
        template <typename T, typename std::enable_if<std::is_floating_point<T>::value, int>::type* = nullptr>
            bool areEqualImpl(const T& lhs, const T& rhs) {
                return (std::abs(rhs - lhs) <= std::numeric_limits<T>::epsilon() * std::max(std::abs(lhs), std::abs(rhs)));
            }
        
        template <typename T>
            bool areEqual(const T &lhs, const T &rhs) {
                return areEqualImpl<T>(lhs,rhs);
            }
        template <typename T>
            bool areNotEqual(const T &lhs, const T &rhs) {
                return !areEqualImpl<T>(lhs,rhs);
            }
        
        
        /* boost related utilities */
        template <std::size_t NumDims>
        struct BuildViewImpl {
            static_assert(NumDims >= 1, "NumDims cannot be < 1");
            boost::detail::multi_array::index_gen<NumDims, NumDims> build() const {
                return BuildViewImpl<NumDims-1>().build()[boost::multi_array_types::index_range()];
            }
        };
        template <>
        struct BuildViewImpl<1> {
            boost::detail::multi_array::index_gen<1,1> build() const {
                return boost::multi_array_types::index_gen()[boost::multi_array_types::index_range()];
            }
        };
        template <std::size_t NumDims>
        boost::detail::multi_array::index_gen<NumDims, NumDims> buildView() {
            return BuildViewImpl<NumDims>().build();
        }

        template <std::size_t NumDims, std::size_t K=NumDims>
            struct BuildIndicesImpl {
                static_assert(NumDims >= 1, "NumDims cannot be < 1");
                const std::array<boost::multi_array_types::index_range, NumDims> &m_ranges;       
                BuildIndicesImpl(const std::array<boost::multi_array_types::index_range, NumDims> &p_ranges): m_ranges(p_ranges) {} 
                boost::detail::multi_array::index_gen<K,K> build() const {
                    return BuildIndicesImpl<NumDims,K-1>(m_ranges).build()[m_ranges[K-1]];
                }
        };
        template <std::size_t NumDims>
            struct BuildIndicesImpl<NumDims,1> {
                const std::array<boost::multi_array_types::index_range, NumDims> &m_ranges;       
                BuildIndicesImpl(const std::array<boost::multi_array_types::index_range, NumDims> &p_ranges): m_ranges(p_ranges) {} 
                boost::detail::multi_array::index_gen<1,1> build() const {
                    return boost::multi_array_types::index_gen()[m_ranges[0]];
                }
        };
        template <std::size_t NumDims>
        boost::detail::multi_array::index_gen<NumDims, NumDims> buildIndices(
                const std::array<boost::multi_array_types::index_range, NumDims> &p_ranges) {
            return BuildIndicesImpl<NumDims>(p_ranges).build();
        }
        
        template <std::size_t NumRanges, std::size_t K=NumRanges>
            struct BuildExtentImpl {
                static_assert(NumRanges >= 1, "NumDims cannot be < 1");
                const std::array<std::size_t,NumRanges>& m_shape;
                BuildExtentImpl(const std::array<std::size_t, NumRanges>& p_shape): m_shape(p_shape) {}
                boost::detail::multi_array::extent_gen<K> build() const {
                    return BuildExtentImpl<NumRanges,K-1>(m_shape).build()[m_shape[K-1]];
                }
            };
        template <std::size_t NumRanges>
            struct BuildExtentImpl<NumRanges,1> {
                const std::array<std::size_t,NumRanges>& m_shape;
                BuildExtentImpl(const std::array<std::size_t, NumRanges>& p_shape): m_shape(p_shape) {}
                boost::detail::multi_array::extent_gen<1> build() const {
                    return boost::multi_array_types::extent_gen()[m_shape[0]];
                }
            };
        template <std::size_t NumRanges>
        boost::detail::multi_array::extent_gen<NumRanges> buildExtents(
                const std::array<std::size_t, NumRanges> &p_shape) {
            return BuildExtentImpl<NumRanges>(p_shape).build();
        }
    }
}


/* quick and dirty fix to allow non namespace dependant operators << for std containers */
namespace std {
    
    template <typename T, std::size_t Dim> 
        std::ostream& operator<<(std::ostream& os, const std::array<T,Dim>& array);
    template <typename T, std::size_t Dim> 
        std::ostream& operator<<(std::ostream& os, const boost::array<T,Dim>& array);

    template <typename T>
        std::ostream& operator<<(std::ostream& os, const std::vector<T>& vector);
    
    template <typename... T>
        std::ostream& operator<<(std::ostream& os, const std::tuple<T...>& tuple);

    std::ostream& operator<<(std::ostream& os, const fftw_iodim& iodim);



    /* Implementation */
    template <typename T, std::size_t Dim>
        std::ostream& operator<<(std::ostream& os, const std::array<T,Dim>& array) {
            os << "[";
            for (std::size_t i = 0; i < Dim-1; i++) 
                os << array[i] << ",";
            os << array[Dim-1];
            os << "]";
            return os;
        }
    template <typename T, std::size_t Dim>
        std::ostream& operator<<(std::ostream& os, const boost::array<T,Dim>& array) {
            os << "[";
            for (std::size_t i = 0; i < Dim-1; i++) 
                os << array[i] << ",";
            os << array[Dim-1];
            os << "]";
            return os;
        }
    template <typename T>
        std::ostream& operator<<(std::ostream& os, const std::vector<T>& vector) {
            os << "[";
            if(!vector.empty()) {
                for (std::size_t i = 0; i < vector.size()-1; i++) 
                    os << vector[i] << ",";
                os << vector[vector.size()-1];
            }
            os << "]";
            return os;
        }
    template <typename... T>
        std::ostream& operator<<(std::ostream& os, const std::tuple<T...>& tuple) {
            hysop::utils::printTuple(os,tuple);
            return os;
        }
}

#endif /* end of include guard: HYSOP_UTILS_H */
