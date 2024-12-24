uint64_t randNum(uint64_t* x)
{
    uint64_t z = ((*x) += UINT64_C(0x9E3779B97F4A7C15));
    z = (z ^ (z >> 30)) * UINT64_C(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94D049BB133111EB);
    return z ^ (z >> 31);
}

double randDouble(uint64_t* x) {
    // Generate a random 64-bit integer
    uint64_t randomInt = randNum(x) % RAND_MAX;

    // Convert to double in the range [0, 1)
    return double(randomInt) / RAND_MAX;
}