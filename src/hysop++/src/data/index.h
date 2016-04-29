
#ifndef HYSOP_INDEX_H
#define HYSOP_INDEX_H

#include <array>
#include <cassert>

namespace hysop {

    template <std::size_t Dim>
        struct Index {
            typedef boost::array<std::ptrdiff_t,Dim> Indices;
            typedef boost::array<std::size_t,Dim>    Dimension;
           
            Index(const Index& idx) = default;
            ~Index() = default;
            
            template <typename DimArray=Dimension, typename IndexArray=Indices>
            Index(const DimArray& p_dim = DimArray{0}, 
                  const Indices &p_ids  = IndexArray{0}) : 
                m_dim(), m_ids(), m_id(0) { 
                    for (std::size_t d = 0; d < Dim; d++) {
                        m_dim[d] = p_dim[d];
                        m_ids[d] = p_ids[d];
                    }
                    initializeId(); 
                } 

            template <typename DimArray, typename IndexArray=Indices>
            Index& reset(const DimArray& p_dim, const IndexArray &p_ids = IndexArray{0}) {
                for (std::size_t d = 0; d < Dim; d++) {
                    m_dim[d] = p_dim[d];
                    m_ids[d] = p_ids[d];
                }
                initializeId();
                return *this;
            }
            
            Index& setIndexToMinusOne() {
                for (std::size_t d=0; d<Dim-1; d++)
                    m_ids[d] = 0;
                m_ids[Dim-1] = -1;
                initializeId();
                return *this;
            }

            std::size_t id() const { return m_id; };
            std::size_t maxId() const { return m_maxId; }
            bool atMaxId() const { return m_id == m_maxId; }
            
            const Indices&   ids() const { return m_ids; };
            const Dimension& dim() const { return m_dim; };

            std::ptrdiff_t  operator[](std::size_t k) const { return m_ids[k]; } 
            std::size_t     operator()() const { return m_id; } 

            virtual void onIndexChange(std::size_t pos, std::ptrdiff_t offset) {};
            virtual void onIndexOverflow(std::size_t pos) {};
            virtual void onIndexUndeflow(std::size_t pos) {};

__attribute__((optimize("unroll-loops")))
            Index& operator++() {    //prefix
                for (int d = Dim-1; d >=0; d--) {
                    m_ids[d]++;
                    if(m_ids[d]==std::ptrdiff_t(m_dim[d])) {
                        m_ids[d]=0;
                        this->onIndexOverflow(d);
                    }
                    else {
                        this->onIndexChange(d, +1);
                        break;
                    }
                }
                m_id++;
                return *this;
            };
__attribute__((optimize("unroll-loops")))
            Index& operator--() {    //prefix
                for (int d = 0; d < Dim; d++) {
                    if(m_ids[d]!=0) {
                        this->onIndexChange(d, -1);
                        m_ids[d]--;
                        break;
                    }
                    else {
                        this->onIndexUndeflow(d);
                    }
                }
                m_id--;
                return *this;
            };

            Index operator++(int) { //postfix
                Index result(*this);
                ++(*this);
                return result;
            };
            Index operator--(int) { //postfix
                Index result(*this);
                --(*this);
                return result;
            };
            
            protected:            
                void initializeId() {
                    m_id = 0;
                    m_maxId = 1;
                    for (std::size_t d = 0; d < Dim-1; d++) {
                        m_id = (m_id + m_ids[d]) * m_dim[d+1]; 
                        m_maxId*=m_dim[d];
                    }
                    m_id += m_ids[Dim-1];
                    m_maxId*=m_dim[Dim-1];
                }

            private:
                Dimension m_dim;
                Indices   m_ids;
                std::size_t m_id, m_maxId;
        };

}

#endif /* end of include guard: HYSOP_INDEX_H */

