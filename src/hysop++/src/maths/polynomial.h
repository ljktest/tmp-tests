
#ifndef HYSOP_POLYNOMIAL_H
#define HYSOP_POLYNOMIAL_H

#include <array>
#include "data/multi_array/multi_array.h"

namespace hysop {
    namespace maths {

        /* Polynomials in dimension Dim with coefficients of type T */
        /* Basic polynomial operations are provided                 */
        /* TODO: Implement fast Nlog(N) multiplication by FFT       */
        /* TODO: Implement polynomial division                      */
        template <typename T, std::size_t Dim>
            class Polynomial;

        template <typename T, std::size_t Dim>
            std::ostream& operator<<(std::ostream& os, const Polynomial<T,Dim>& poly);

        template <typename T, std::size_t Dim>
            class Polynomial {
                public:
                    /* constructors, destructors & operator= */
                    Polynomial()                                      = default;
                    Polynomial(const Polynomial&  p_other)            = default;
                    Polynomial(      Polynomial&& p_other)            = default;
                    Polynomial& operator=(const Polynomial& p_other)  = default;
                    Polynomial& operator=(      Polynomial&& p_other) = default;
                    ~Polynomial()                                     = default;

                    Polynomial(const std::array<std::size_t,Dim>& p_order);
                    explicit Polynomial(const hysop::multi_array<T,Dim>&  p_coeffs);
                    explicit Polynomial(      hysop::multi_array<T,Dim>&& p_coeffs);

                    template <typename U>
                    explicit Polynomial(const Polynomial<U,Dim>& p_other);
                    
                    /* accessors */
                    const hysop::multi_array<T,Dim>&    coefficients() const;
                          hysop::multi_array<T,Dim>&    coefficients();
                    const std::array<std::size_t,Dim>&  order()        const;
                          std::array<std::size_t,Dim>   shape()        const; /* == order + 1 */

                    /* mutators */
                    Polynomial& reshape(const std::array<std::size_t,Dim>& p_shape);
                    Polynomial& setOrder(const std::array<std::size_t,Dim>& p_order);
            
                    Polynomial& applyToCoefficients(const std::function<void(T&)>& func);
                    Polynomial& applyToCoefficients(const std::function<void(T&, const Index<Dim>&)>& func);
                    
                    /* apply func(T&, const Index<Dim>&, farg0, fargs...) to all coefficients */
                    template <typename Functor, typename Arg0, typename... Args> 
                        Polynomial& applyToCoefficients(const Functor& func, Arg0&& farg0, Args&&... fargs);
                    
                    /* elementwise access to coefficients */
                    const T& operator[](const Index<Dim> &p_id) const;
                          T& operator[](const Index<Dim> &p_id);
                    const T& operator[](std::size_t k) const;
                          T& operator[](std::size_t k);

                    /* polynomial function evaluation with arbitrary type */
                    template <typename U1, typename... U, typename R = typename std::common_type<T,U1,U...>::type> 
                        R operator()(const U1& x1, const U&... xs) const;
                    template <typename U, typename R=typename std::common_type<T,U>::type> 
                        R operator()(const std::array<U,Dim> &x) const;
                    template <typename U, typename R=typename std::common_type<T,U>::type> 
                        R operator()(const U* x) const;

                    /* basic elementwise operations */
                    Polynomial& operator+=(const Polynomial& p_other);
                    Polynomial& operator-=(const Polynomial& p_other);

                    Polynomial& operator*=(const T& p_val);
                    Polynomial& operator/=(const T& p_val);
                    Polynomial& operator%=(const T& p_val);
                    
                    /* polynomial multiplication and division */
                    Polynomial& operator*=(const Polynomial& p_other);
                    Polynomial& operator/=(const Polynomial& p_other);
                
                    /* integral and derivatives */
                    Polynomial& integrate    (std::size_t dim, int order);
                    Polynomial& differentiate(std::size_t dim, int order);

                    template <typename I> typename std::enable_if<std::is_integral<I>::value, Polynomial&>::type                        
                        differentiate(const std::array<I,Dim>& order);
                    template <typename I> typename std::enable_if<std::is_integral<I>::value, Polynomial&>::type                        
                        integrate(const std::array<I,Dim>& order);

                    template <typename I> typename std::enable_if<std::is_integral<I>::value, Polynomial&>::type
                    operator >>=(const std::array<I,Dim>& order);
                    template <typename I> typename std::enable_if<std::is_integral<I>::value, Polynomial&>::type
                    operator <<=(const std::array<I,Dim>& order);

                    /* comparisson operators - uses near equality if T is a floating point type */
                    bool operator==(const Polynomial& other);
                    bool operator!=(const Polynomial& other);

                    /* misc */
                    std::string toString(unsigned int p_precision=2, unsigned int p_width=6) const;

                protected:
                    /* misc */
                    template <std::size_t D>
                    std::string toStringImpl(const T& p_coeff, unsigned int p_precision, unsigned int p_width, bool p_begin, bool p_end) const {
                        std::stringstream ss;
                        ss << (p_begin ? "" : " ") << std::fixed << std::showpos << std::setprecision(p_precision) << p_coeff;
                        return ss.str();
                    }
                    template <std::size_t D, typename ArrayView, std::size_t K=Dim-D>
                        std::string toStringImpl(const ArrayView& p_view, unsigned int p_precision, unsigned int p_width, 
                                bool=false, bool=false) const {
                            static const char varNames[3] = { 'z','y','x' };
                            static const int offset = (Dim==1 ? 2 : (Dim==2 ? 1 : (Dim==3 ? 0 : -1)));
                            static const char delimiters[3][2] = { {'[',']'},
                                                                   {'{','}'},
                                                                   {'(',')'} };


                            std::string str;
                            for (std::ptrdiff_t k=m_coeffs.shape()[K]-1; k>=0; k--) {
                                std::string localStr = toStringImpl<D-1>(
                                        p_view[k], p_precision, p_width, k==std::ptrdiff_t(m_coeffs.shape()[K]-1), k==0);
                                if(localStr!="") {
                                    if(D>1)
                                        str += delimiters[D%3][0];
                                    str += localStr;
                                    if(D>1)
                                        str += delimiters[D%3][1];
                                    
                                    std::string varName;
                                    if(Dim<=3)
                                        varName = varNames[K+offset];
                                    else 
                                        varName = "x_"+std::to_string(D);
                                    if(k==0)
                                        ;
                                    else if(k==1)
                                        str += varName;
                                    else
                                        str += varName + "^" + std::to_string(k);
                                    if(k>0 && D>1)
                                        str += " + ";
                                }
                            }
                            return str;
                    }

                    /* static members */
                    static std::array<std::size_t,Dim> orderFromShape(const std::array<std::size_t,Dim>& p_shape);
                    static std::array<std::size_t,Dim> shapeFromOrder(const std::array<std::size_t,Dim>& p_order);
               
                public:
                    template <typename X>
                        struct PolynomIndex : public Index<Dim> {

                            public:
                                using typename Index<Dim>::Dimension;
                                using typename Index<Dim>::Indices;
                            public:
                                template <typename DimArray=typename Index<Dim>::Dimension, typename IndexArray=typename Index<Dim>::Indices>
                                PolynomIndex(std::array<X,Dim> p_spaceVar,
                                        const DimArray&   p_dim = Dimension{0}, 
                                        const IndexArray& p_ids = Indices{0});
                                const std::array<X,Dim>& spaceVariable() const; /* returns {X[0],...,X[Dim-1]} */
                                                               X value() const; /* returns X[0]^id[0] * X1^id[1] * ... * X[Dim-1]^id[Dim-1]*/
                            protected:
                                void initialize();
                                virtual void onIndexChange   (std::size_t p_pos, std::ptrdiff_t p_offset) final override;
                                virtual void onIndexOverflow (std::size_t p_pos) final override;

                            protected:
                                const std::array<X,Dim> m_spaceVar;
                                std::array<X,Dim> m_powers; 
                                X m_value;
                        };

                protected:
                    hysop::multi_array<T,Dim>   m_coeffs;
                    std::array<std::size_t,Dim> m_order;
            };

        /* unary operations */
        template <typename T, std::size_t Dim>
            Polynomial<T,Dim> operator+(const Polynomial<T,Dim>& poly);
        template <typename T, std::size_t Dim>
            Polynomial<T,Dim> operator-(const Polynomial<T,Dim>& poly);

        /* basic operations */
        template <typename T1, typename T2, std::size_t Dim, typename T = typename std::common_type<T1,T2>::type>
            Polynomial<T,Dim> operator+(const Polynomial<T1,Dim>& lhs, const Polynomial<T2,Dim>& rhs);
        template <typename T1, typename T2, std::size_t Dim, typename T = typename std::common_type<T1,T2>::type>
            Polynomial<T,Dim> operator-(const Polynomial<T1,Dim>& lhs, const Polynomial<T2,Dim>& rhs);
        template <typename T1, typename T2, std::size_t Dim, typename T = typename std::common_type<T1,T2>::type>
            Polynomial<T,Dim> operator*(const Polynomial<T1,Dim>& lhs, const Polynomial<T2,Dim>& rhs);
        template <typename T1, typename T2, std::size_t Dim, typename T = typename std::common_type<T1,T2>::type>
            Polynomial<T,Dim> operator/(const Polynomial<T1,Dim>& lhs, const Polynomial<T2,Dim>& rhs);


        /* tensor product of polynomials */
        template <typename T1, typename T2, std::size_t Dim1, std::size_t Dim2, 
                  typename T=typename std::common_type<T1,T2>::type, std::size_t Dim=Dim1+Dim2>
            Polynomial<T,Dim> operator|(const Polynomial<T1,Dim1>& lhs, const Polynomial<T2,Dim2>& rhs);
        

        /* integral and derivatives */
        template <typename T, std::size_t Dim, typename I>
            typename std::enable_if<std::is_integral<I>::value, Polynomial<T,Dim>>::type
            operator<<(const Polynomial<T,Dim>& lhs, const std::array<I,Dim>& k);
        template <typename T, std::size_t Dim, typename I>
            typename std::enable_if<std::is_integral<I>::value, Polynomial<T,Dim>>::type
            operator>>(const Polynomial<T,Dim>& lhs, const std::array<I,Dim>& k);

        template <typename T, std::size_t Dim, typename I>
            typename std::enable_if<std::is_integral<I>::value, Polynomial<T,Dim>>::type
            operator<<(Polynomial<T,Dim>&& lhs, const std::array<I,Dim>& k);
        template <typename T, std::size_t Dim, typename I>
            typename std::enable_if<std::is_integral<I>::value, Polynomial<T,Dim>>::type
            operator>>(Polynomial<T,Dim>&& lhs, const std::array<I,Dim>& k);
        


          /********************/
         /** IMPLEMENTATION **/
        /********************/
        
        /* static members */
        template <typename T, std::size_t Dim>
            std::array<std::size_t,Dim> Polynomial<T,Dim>::orderFromShape(const std::array<std::size_t,Dim>& p_shape) {
                std::array<std::size_t,Dim> order;
                for (std::size_t d = 0; d < Dim; d++)
                    order[d] = p_shape[d]-1;
                return order;
            }

        template <typename T, std::size_t Dim>
            std::array<std::size_t,Dim> Polynomial<T,Dim>::shapeFromOrder(const std::array<std::size_t,Dim>& p_order) {
                std::array<std::size_t,Dim> shape;
                for (std::size_t d = 0; d < Dim; d++)
                    shape[d] = p_order[d]+1;
                return shape;
            }

        /* constructors, destructors & operator= */
        template <typename T, std::size_t Dim>
        Polynomial<T,Dim>::Polynomial(const std::array<std::size_t,Dim>& p_shape) :
            m_coeffs(), m_order() {
                this->reshape(p_shape);
        }
        
        template <typename T, std::size_t Dim>
            Polynomial<T,Dim>::Polynomial(const hysop::multi_array<T,Dim>&  p_coeffs) :
                m_coeffs(p_coeffs), m_order(orderFromShape(m_coeffs.shape())) {
        }
        
        template <typename T, std::size_t Dim>
            Polynomial<T,Dim>::Polynomial(hysop::multi_array<T,Dim>&& p_coeffs) :
                m_coeffs(std::move(p_coeffs)), m_order(orderFromShape(m_coeffs.shape())) {
        }
                    
        template <typename T, std::size_t Dim>
        template <typename U>
            Polynomial<T,Dim>::Polynomial(const Polynomial<U,Dim>& p_other) {
                this->reshape(p_other.shape());
                for (std::size_t k=0; k < m_coeffs.num_elements(); k++)
                    m_coeffs.data()[k] = static_cast<T>(p_other.data()[k]);
        }
                    
        /* accessors */
        template <typename T, std::size_t Dim>
        const hysop::multi_array<T,Dim>& Polynomial<T,Dim>::coefficients() const {
            return m_coeffs;
        }
        template <typename T, std::size_t Dim>
        hysop::multi_array<T,Dim>& Polynomial<T,Dim>::coefficients() {
            return m_coeffs;
        }
        template <typename T, std::size_t Dim>
        const std::array<std::size_t,Dim>&  Polynomial<T,Dim>::order() const {
            return m_order;
        }
        template <typename T, std::size_t Dim>
        std::array<std::size_t,Dim> Polynomial<T,Dim>::shape() const {
            return m_coeffs.shape();
        }
                    
        /* mutators */
        template <typename T, std::size_t Dim>
        Polynomial<T,Dim>& Polynomial<T,Dim>::reshape(const std::array<std::size_t,Dim>& p_shape) {
            m_order = orderFromShape(p_shape);
            m_coeffs.reshape(p_shape);
            return *this;
        }
        
        template <typename T, std::size_t Dim>
        Polynomial<T,Dim>& Polynomial<T,Dim>::setOrder(const std::array<std::size_t,Dim>& p_order) {
            m_order = p_order;
            m_coeffs.reshape(shapeFromOrder(p_order));
            return *this;
        }
            

        template <typename T, std::size_t Dim>
        Polynomial<T,Dim>& Polynomial<T,Dim>::applyToCoefficients(const std::function<void(T&)>& func) {
            m_coeffs.apply(func);
            return *this;
        }
        template <typename T, std::size_t Dim>
        Polynomial<T,Dim>& Polynomial<T,Dim>::applyToCoefficients(const std::function<void(T&, const Index<Dim>&)>& func) {
            m_coeffs.apply(func);
            return *this;
        }

        /* apply func(T&, const Index<Dim>&, farg0, fargs...) on all coefficients */
        template <typename T, std::size_t Dim>
        template <typename Functor, typename Arg0, typename... Args> 
        Polynomial<T,Dim>& Polynomial<T,Dim>::applyToCoefficients(const Functor& func, Arg0&& farg0, Args&&... fargs) {
            m_coeffs.apply(func, farg0, fargs...);
            return *this;
        }

        /* access to coefficients */
        template <typename T, std::size_t Dim>
        const T& Polynomial<T,Dim>::operator[](std::size_t k) const {
            return m_coeffs.data()[k];
        }
        template <typename T, std::size_t Dim>
        T& Polynomial<T,Dim>::operator[](std::size_t k) {
            return m_coeffs.data()[k];
        }
        template <typename T, std::size_t Dim>
        const T& Polynomial<T,Dim>::operator[](const Index<Dim> &p_id) const {
            return m_coeffs.data()[p_id.id()];
        }

        template <typename T, std::size_t Dim>
        T& Polynomial<T,Dim>::operator[](const Index<Dim> &p_id) {
            return m_coeffs.data()[p_id.id()];
        }

        /* polynomial evaluation */
        template <typename T, std::size_t Dim>
        template <typename U1, typename... U, typename R> 
            R Polynomial<T,Dim>::operator()(const U1& x1, const U&... xs) const {
                return this->operator()(std::array<R,Dim>{x1,xs...});
        }
        template <typename T, std::size_t Dim>
        template <typename U, typename R>
            R Polynomial<T,Dim>::operator()(const U* p_x) const {
                return this->operator()(std::array<U,Dim>(p_x));
        }
        template <typename T, std::size_t Dim>
        template <typename U, typename R>
            R Polynomial<T,Dim>::operator()(const std::array<U,Dim> &p_x) const {
                /* compute result */
                R res = R(0);
                const T* coeffs = m_coeffs.data();
                PolynomIndex<U> idx(p_x, this->shape());
                while(!idx.atMaxId()) {
                    res += coeffs[idx()]*idx.value();
                    ++idx;
                }
                return res;
        }
                    
        template <typename T, std::size_t Dim> 
        std::string Polynomial<T,Dim>::toString(unsigned int p_precision, unsigned int p_width) const {
            return toStringImpl<Dim>(m_coeffs, p_precision, p_width); 
        }
        
        /* struct PolynomIndex */
        template <typename T, std::size_t Dim> 
            template <typename X> 
                template <typename DimArray, typename IndexArray>
            Polynomial<T,Dim>::PolynomIndex<X>::
            PolynomIndex(std::array<X,Dim> p_spaceVar, const DimArray& p_dim, const IndexArray& p_ids):
                Index<Dim>(p_dim, p_ids), m_spaceVar(p_spaceVar), m_powers{0}, m_value(0) {
                    this->initialize();
            }
        template <typename T, std::size_t Dim> 
            template <typename X> 
            void Polynomial<T,Dim>::PolynomIndex<X>::initialize() {
                m_value = X(1);
                for (std::size_t d=0; d<Dim; d++) {
                    X power = std::pow(m_spaceVar[d],this->operator[](d));
                    m_powers[d] = power;
                    m_value *= power; 
                }
            }
        template <typename T, std::size_t Dim> 
            template <typename X> 
            const std::array<X,Dim>& Polynomial<T,Dim>::PolynomIndex<X>::spaceVariable() const {
                return m_spaceVar;
            }
        template <typename T, std::size_t Dim> 
            template <typename X> 
            X Polynomial<T,Dim>::PolynomIndex<X>::value() const {
                return m_value;
            }
        template <typename T, std::size_t Dim> 
            template <typename X> 
            void Polynomial<T,Dim>::PolynomIndex<X>::onIndexChange(std::size_t p_pos, std::ptrdiff_t p_offset) {
                assert(p_offset == 1);
                m_powers[p_pos] = m_powers[p_pos]*m_spaceVar[p_pos];
                m_value *= m_spaceVar[p_pos];
            }
        template <typename T, std::size_t Dim> 
            template <typename X> 
            void Polynomial<T,Dim>::PolynomIndex<X>::onIndexOverflow (std::size_t p_pos) {
                m_powers[p_pos] = X(1);
                m_value = X(1);
                for (std::size_t d=0; d < Dim; d++) 
                    m_value *= m_powers[d];
            }
        
        template <typename T, std::size_t Dim>
            std::ostream& operator<<(std::ostream& os, const Polynomial<T,Dim>& poly) {
                os << poly.toString();
                return os;
            }

    } /* end of namespace maths */
} /* end of namesapce hysop */


#endif /* end of include guard: HYSOP_POLYNOMIAL_H */
