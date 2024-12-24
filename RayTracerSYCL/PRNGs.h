uint64_t shuffle_table[4];
// The actual algorithm
uint64_t Xorshift(void)
{
    uint64_t s1 = shuffle_table[0];
    uint64_t s0 = shuffle_table[1];
    uint64_t result = s0 + s1;
    shuffle_table[0] = s0;
    s1 ^= s1 << 23;
    shuffle_table[1] = s1 ^ s0 ^ (s1 >> 18) ^ (s0 >> 5);
    return result;
}

class SplitMix64 {
public:

    uint64_t x;

    SplitMix64(uint64_t seed) : x(seed) {}

    uint64_t randNum()
    {
        uint64_t z = (x += UINT64_C(0x9E3779B97F4A7C15));
        z = (z ^ (z >> 30)) * UINT64_C(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)) * UINT64_C(0x94D049BB133111EB);
        return z ^ (z >> 31);
    }

    void SeedSplitMix64(uint64_t seed) {
        x = seed;
    }

    double randDouble() {
        // Generate a random 64-bit integer
        uint64_t randomInt = randNum() % RAND_MAX;

        // Convert to double in the range [0, 1)
        return double(randomInt) / RAND_MAX;
    }

};