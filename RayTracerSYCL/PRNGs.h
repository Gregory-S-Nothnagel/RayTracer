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

uint32_t randNum32(uint32_t* x)
{
    uint32_t z = ((*x) += 0x9E3779B9); // Add a large constant (Golden Ratio for 32-bit)
    z = (z ^ (z >> 16)) * 0x85EBCA6B;  // Mix using a 32-bit prime constant
    z = (z ^ (z >> 13)) * 0xC2B2AE35;  // Another mix with a different constant
    return z ^ (z >> 16);              // Final mixing step
}


float randFloat(uint32_t* x) {
    // Generate a random 32-bit integer (cast to float directly to avoid double)
    uint32_t randomInt = randNum32(x);

    // Convert to float in the range [0, 1)
    return float(randomInt) / 4294967296.0f; // Use 2^32 as divisor
}