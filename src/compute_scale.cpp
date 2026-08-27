#include "Rtatami.h"

#include <cmath>
#include <algorithm>
#include <optional>
#include <type_traits>

#include "Rcpp.h"

//[[Rcpp::export(rng=false)]]
SEXP set_executor(SEXP ptr) {
    Rtatami::set_executor(ptr);
    return R_NilValue;
}

void compute_scale_direct(const tatami::NumericMatrix& mat, const double* cptr, double* optr, int nthreads) {
    const auto NR = mat.nrow();
    const auto NC = mat.ncol();

    if (mat.is_sparse()) {
        tatami::parallelize([&](size_t, int start, int len) -> void {
            tatami::Options opt;
            opt.sparse_extract_index = false;
            auto ext = tatami::consecutive_extractor<true>(mat, false, start, len, opt);
            std::vector<double> vbuffer(NR);

            for (int c = start, end = start + len; c < end; ++c) {
                auto range = ext->fetch(vbuffer.data(), NULL);
                double center = cptr[c];

                double tmp = 0;
                for (int i = 0; i < range.number; ++i) {
                    double diff = range.value[i] - center;
                    tmp += diff * diff;
                }

                tmp += (NR - range.number) * center * center;
                optr[c] = std::sqrt(tmp / static_cast<double>(NR - 1));
            }
        }, NC, nthreads);

    } else {
        tatami::parallelize([&](size_t, int start, int len) -> void {
            auto ext = tatami::consecutive_extractor<false>(mat, false, start, len);
            std::vector<double> buffer(NR);
            for (int c = start, end = start + len; c < end; ++c) {
                auto ptr = ext->fetch(buffer.data());
                double center = cptr[c];

                double tmp = 0;
                for (int r = 0; r < NR; ++r) {
                    double diff = ptr[r] - center;
                    tmp += diff * diff;
                }
                optr[c] = std::sqrt(tmp / static_cast<double>(NR - 1));
            }
        }, NC, nthreads);
    }
}

void compute_scale_running(const tatami::NumericMatrix& mat, const double* cptr, double* optr, int nthreads) {
    const bool do_parallel = nthreads > 1;
    std::optional<std::vector<std::optional<std::vector<double> > > > tmp_sums;
    if (do_parallel) {
        tmp_sums.emplace(sanisizer::cast<decltype(tmp_sums->size())>(nthreads - 1));
    }

    const bool is_sparse = mat.is_sparse();
    const auto NR = mat.nrow();
    const auto NC = mat.ncol();

    const int nused = tatami::parallelize([&](int t, int start, int len) -> void {
        std::optional<std::vector<double> > tmp_sum;
        double* outptr;
        if (t > 0) {
            tmp_sum.emplace(tatami::cast_Index_to_container_size<std::vector<double> >(NC));
            outptr = tmp_sum->data();
        } else {
            // We assume that this is already zeroed by Rcpp.
            outptr = optr;
        }

        if (is_sparse) { 
            auto ext = tatami::consecutive_extractor<true>(mat, true, start, len);
            std::vector<double> vbuffer(NC);
            std::vector<int> ibuffer(NC);
            std::vector<int> nonzeros(NC);

            for (int r = 0; r < len; ++r) {
                auto range = ext->fetch(vbuffer.data(), ibuffer.data());
                for (int i = 0; i < range.number; ++i) {
                    double diff = range.value[i] - cptr[range.index[i]];
                    outptr[range.index[i]] += diff * diff;
                    ++nonzeros[range.index[i]];
                }
            }

            for (int c = 0; c < NC; ++c) {
                outptr[c] += cptr[c] * cptr[c] * (len - nonzeros[c]);
            }

        } else {
            auto ext = tatami::consecutive_extractor<false>(mat, true, start, len);
            std::vector<double> buffer(NC);

            for (int r = 0; r < len; ++r) {
                auto ptr = ext->fetch(buffer.data());
                for (int c = 0; c < NC; ++c) {
                    double diff = ptr[c] - cptr[c];
                    outptr[c] += diff * diff;
                }
            }
        }

        if (t > 0) {
            (*tmp_sums)[t - 1] = std::move(tmp_sum);
        }
    }, NR, nthreads);

    if (do_parallel) {
        for (int u = 1; u < nused; ++u) {
            const auto& cursums = *((*tmp_sums)[u - 1]);
            for (int c = 0; c < NC; ++c) {
                optr[c] += cursums[c];
            }
        }
    }

    for (int c = 0; c < NC; ++c) {
        optr[c] = std::sqrt(optr[c] / static_cast<double>(NR - 1));
    }
}

// [[Rcpp::export(rng=false)]]
Rcpp::NumericVector compute_scale(Rcpp::RObject mat, Rcpp::NumericVector centers, int nthreads) {
    Rtatami::BoundNumericPointer bound(mat);
    const auto& ptr = bound->ptr;
    auto NR = ptr->nrow();
    auto NC = ptr->ncol();

    if (NC != static_cast<decltype(NC)>(centers.size())) {
        throw std::runtime_error("'centers' should be equal to the number of columns in 'mat'");
    }

    Rcpp::NumericVector output(NC);
    if (NR <= 1) {
        std::fill(output.begin(), output.end(), std::numeric_limits<double>::quiet_NaN());
        return output;
    }

    if (ptr->prefer_rows()) {
        compute_scale_running(*ptr, centers.begin(), output.begin(), nthreads);
    } else {
        compute_scale_direct(*ptr, centers.begin(), output.begin(), nthreads);
    }
    return output;
}
