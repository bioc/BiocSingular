#' @importFrom DelayedArray DelayedArray 
#' @importFrom ScaledMatrix ScaledMatrix
#' @importFrom BiocParallel SerialParam bpnworkers
standardize_matrix <- function(x, center=FALSE, scale=FALSE, deferred=FALSE, BPPARAM=SerialParam())
# Creates a deferred or delayed centered and scaled matrix.
# The two choices have different implications for speed and accuracy.
{
    stats <- .compute_center_and_scale(x, center, scale, bpnworkers(BPPARAM))
    center <- stats$center
    scale <- stats$scale
    if (!is.null(scale)) {
        scale[scale == 0] <- 1
    }

    if (deferred) {
        if (bpnworkers(BPPARAM)==1L) {
            original <- x # exploit original (non-DA) matrix mult functions.
        } else {
            original <- DelayedArray(x) # exploit parallelization for DAs.
        }
        X <- ScaledMatrix(original, center=center, scale=scale)

    } else {
        X <- DelayedArray(x)
        if (!is.null(center)) {
            X <- sweep(X, 2, center, "-") 
        }
        if (!is.null(scale)) {
            X <- sweep(X, 2, scale, "/")
        }
    }
    return(X)
}

#' @useDynLib BiocSingular
#' @importFrom Rcpp sourceCpp
#' @importFrom beachmat initializeCpp tatami.sums tatami.variances
.compute_center_and_scale <- function(x, center, scale, nthreads) {
    if (isTRUE(center) && isTRUE(scale)) {
        ptr <- initializeCpp(x)
        out <- tatami.variances(ptr, row=FALSE, num.threads=nthreads)
        center <- out$mean
        scale <- sqrt(out$variance)
    }

    if (is.logical(center)) {
        if (center) {
            ptr <- initializeCpp(x)
            center <- tatami.sums(ptr, row=FALSE, num.threads=nthreads) / nrow(x)
        } else {
            center <- NULL
        }
    }

    if (is.logical(scale)) {
        if (scale) {
            ptr <- initializeCpp(x)
            tmp_center <- center
            if (is.null(tmp_center)) {
                tmp_center <- numeric(ncol(x))
            }
            scale <- compute_scale(ptr, tmp_center, nthreads)
        } else {
            scale <- NULL
        }
    }

    list(center=center, scale=scale) 
}

standardize_output_SVD <- function(res, x) 
# Provide a common standard output for all SVD functions.
{
    res$d <- as.numeric(res$d)
    res$u <- as.matrix(res$u)
    rownames(res$u) <- rownames(x)
    res$v <- as.matrix(res$v)
    rownames(res$v) <- colnames(x)
    res[c("d", "u", "v")]
}
