"""Numerical kernels for classical time-series decomposition and tests."""

from std.sys import simd_width_of


comptime W = simd_width_of[DType.float64]()
comptime Ptr = UnsafePointer[Float64, AnyOrigin[mut=True]]


def p(addr: Int) -> Ptr:
    return Ptr(unsafe_from_address=addr)


def dot(a: Ptr, b: Ptr, n: Int) -> Float64:
    var acc = SIMD[DType.float64, W](0.0)
    var i = 0
    while i + W <= n:
        acc += a.load[width=W](i) * b.load[width=W](i)
        i += W
    var total = acc.reduce_add()
    while i < n:
        total += a[i] * b[i]
        i += 1
    return total


def fill(dst: Ptr, start: Int, stop: Int, value: Float64):
    var values = SIMD[DType.float64, W](value)
    var i = start
    while i + W <= stop:
        dst.store(i, values)
        i += W
    while i < stop:
        dst[i] = value
        i += 1


def convolution_range(
    x: Ptr,
    filt: Ptr,
    dst: Ptr,
    cols: Int,
    m: Int,
    dst_offset: Int,
    start: Int,
    stop: Int,
):
    var q = start
    while q + W <= stop:
        var acc = SIMD[DType.float64, W](0.0)
        for j in range(m):
            acc += (
                x.load[width=W](q + j * cols)
                * SIMD[DType.float64, W](filt[m - 1 - j])
            )
        dst.store(dst_offset + q, acc)
        q += W
    while q < stop:
        var total = 0.0
        for j in range(m):
            total += x[q + j * cols] * filt[m - 1 - j]
        dst[dst_offset + q] = total
        q += 1


@export("mts_acovf")
def mts_acovf(
    x_addr: Int,
    n: Int,
    nlags: Int,
    demean: Int,
    adjusted: Int,
    dst_addr: Int,
) abi("C"):
    var x = p(x_addr)
    var dst = p(dst_addr)
    var mean = 0.0
    if demean != 0:
        mean = dot(x, p(x_addr), 0)
        var sum_vec = SIMD[DType.float64, W](0.0)
        var si = 0
        while si + W <= n:
            sum_vec += x.load[width=W](si)
            si += W
        mean = sum_vec.reduce_add()
        while si < n:
            mean += x[si]
            si += 1
        mean /= Float64(n)

    for lag in range(nlags + 1):
        var acc = SIMD[DType.float64, W](0.0)
        var i = 0
        var count = n - lag
        while i + W <= count:
            var left = x.load[width=W](i + lag) - SIMD[DType.float64, W](mean)
            var right = x.load[width=W](i) - SIMD[DType.float64, W](mean)
            acc += left * right
            i += W
        var total = acc.reduce_add()
        while i < count:
            total += (x[i + lag] - mean) * (x[i] - mean)
            i += 1
        var divisor = n
        if adjusted != 0:
            divisor = count
        dst[lag] = total / Float64(divisor)


@export("mts_ccovf")
def mts_ccovf(
    x_addr: Int,
    y_addr: Int,
    n: Int,
    nlags: Int,
    demean: Int,
    adjusted: Int,
    dst_addr: Int,
) abi("C"):
    var x = p(x_addr)
    var y = p(y_addr)
    var dst = p(dst_addr)
    var xmean = 0.0
    var ymean = 0.0
    if demean != 0:
        for i in range(n):
            xmean += x[i]
            ymean += y[i]
        xmean /= Float64(n)
        ymean /= Float64(n)
    for lag in range(nlags):
        var acc = SIMD[DType.float64, W](0.0)
        var i = 0
        var count = n - lag
        while i + W <= count:
            var left = x.load[width=W](i + lag) - SIMD[DType.float64, W](xmean)
            var right = y.load[width=W](i) - SIMD[DType.float64, W](ymean)
            acc += left * right
            i += W
        var total = acc.reduce_add()
        while i < count:
            total += (x[i + lag] - xmean) * (y[i] - ymean)
            i += 1
        var divisor = n
        if adjusted != 0:
            divisor = count
        dst[lag] = total / Float64(divisor)


@export("mts_q_stat")
def mts_q_stat(x_addr: Int, m: Int, nobs: Int, dst_addr: Int) abi("C"):
    var x = p(x_addr)
    var dst = p(dst_addr)
    var cumulative = 0.0
    # Convert before multiplying so a large, user-supplied nobs cannot overflow
    # Int even though the final calculation is floating point.
    var factor = Float64(nobs) * (Float64(nobs) + 2.0)
    for i in range(m):
        cumulative += x[i] * x[i] / Float64(nobs - i - 1)
        dst[i] = factor * cumulative


@export("mts_convolution")
def mts_convolution(
    x_addr: Int,
    n: Int,
    cols: Int,
    filt_addr: Int,
    m: Int,
    nsides: Int,
    nan: Float64,
    dst_addr: Int,
) abi("C"):
    var x = p(x_addr)
    var filt = p(filt_addr)
    var dst = p(dst_addr)
    var head = m - 1
    if nsides == 2:
        head = (m + 1) // 2 - 1
    var valid = n - m + 1
    var valid_values = valid * cols
    var dst_offset = head * cols
    fill(dst, 0, dst_offset, nan)
    fill(dst, dst_offset + valid_values, n * cols, nan)
    convolution_range(x, filt, dst, cols, m, dst_offset, 0, valid_values)


@export("mts_seasonal_mean")
def mts_seasonal_mean(
    x_addr: Int,
    n: Int,
    cols: Int,
    period: Int,
    nan: Float64,
    dst_addr: Int,
) abi("C"):
    var x = p(x_addr)
    var dst = p(dst_addr)
    for phase in range(period):
        for c in range(cols):
            var total = 0.0
            var count = 0
            var i = phase
            while i < n:
                var value = x[i * cols + c]
                if value == value:
                    total += value
                    count += 1
                i += period
            if count == 0:
                dst[phase * cols + c] = nan
            else:
                dst[phase * cols + c] = total / Float64(count)


@export("mts_seasonal_mean_detrended")
def mts_seasonal_mean_detrended(
    x_addr: Int,
    trend_addr: Int,
    n: Int,
    cols: Int,
    period: Int,
    multiplicative: Int,
    nan: Float64,
    dst_addr: Int,
    scratch_addr: Int,
) abi("C"):
    var x = p(x_addr)
    var trend = p(trend_addr)
    var dst = p(dst_addr)
    var scratch = p(scratch_addr)
    var cycle = period * cols
    fill(dst, 0, cycle, 0.0)
    fill(scratch, 0, cycle, 0.0)
    var total_values = n * cols
    var base = 0
    while base < total_values:
        var block = min(cycle, total_values - base)
        for q in range(block):
            var value = x[base + q] - trend[base + q]
            if multiplicative != 0:
                value = x[base + q] / trend[base + q]
            if value == value:
                dst[q] += value
                scratch[q] += 1.0
        base += cycle
    for q in range(cycle):
        if scratch[q] == 0.0:
            dst[q] = nan
        else:
            dst[q] /= scratch[q]


@export("mts_seasonal_resid")
def mts_seasonal_resid(
    x_addr: Int,
    trend_addr: Int,
    period_mean_addr: Int,
    n: Int,
    cols: Int,
    period: Int,
    multiplicative: Int,
    seasonal_addr: Int,
    resid_addr: Int,
) abi("C"):
    var x = p(x_addr)
    var trend = p(trend_addr)
    var period_mean = p(period_mean_addr)
    var seasonal = p(seasonal_addr)
    var resid = p(resid_addr)
    var cycle = period * cols
    var total_values = n * cols
    var base = 0
    while base < total_values:
        var block = min(cycle, total_values - base)
        var q = 0
        while q + W <= block:
            var phase = period_mean.load[width=W](q)
            var observed = x.load[width=W](base + q)
            var trend_value = trend.load[width=W](base + q)
            seasonal.store(base + q, phase)
            if multiplicative != 0:
                resid.store(base + q, observed / trend_value / phase)
            else:
                resid.store(base + q, observed - trend_value - phase)
            q += W
        while q < block:
            var phase = period_mean[q]
            seasonal[base + q] = phase
            if multiplicative != 0:
                resid[base + q] = x[base + q] / trend[base + q] / phase
            else:
                resid[base + q] = x[base + q] - trend[base + q] - phase
            q += 1
        base += cycle


@export("mts_kpss_moments")
def mts_kpss_moments(
    residual_addr: Int,
    n: Int,
    lags: Int,
    dst_addr: Int,
) abi("C"):
    var residual = p(residual_addr)
    var dst = p(dst_addr)
    var running = 0.0
    var eta_sum = 0.0
    for i in range(n):
        running += residual[i]
        eta_sum += running * running
    var long_run = dot(residual, residual, n)
    for lag in range(1, lags + 1):
        var cross = dot(residual + lag, residual, n - lag)
        long_run += 2.0 * cross * (1.0 - Float64(lag) / Float64(lags + 1))
    dst[0] = eta_sum / Float64(n * n)
    dst[1] = long_run / Float64(n)


@export("mts_ols_moments")
def mts_ols_moments(
    x_addr: Int,
    y_addr: Int,
    n: Int,
    k: Int,
    gram_addr: Int,
    rhs_addr: Int,
) abi("C"):
    var x = p(x_addr)
    var y = p(y_addr)
    var gram = p(gram_addr)
    var rhs = p(rhs_addr)
    for i in range(k * k):
        gram[i] = 0.0
    for i in range(k):
        rhs[i] = 0.0
    for row in range(n):
        var row_ptr = x + row * k
        for i in range(k):
            var xi = row_ptr[i]
            rhs[i] += xi * y[row]
            for j in range(i + 1):
                gram[i * k + j] += xi * row_ptr[j]
    for i in range(k):
        for j in range(i + 1, k):
            gram[i * k + j] = gram[j * k + i]
