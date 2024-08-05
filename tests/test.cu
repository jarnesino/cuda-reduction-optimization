#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"

TEST_SUITE("suite") {
    TEST_CASE("test") {
        CHECK(1 == 1);
    }
}

int main() {
    doctest::Context context;

    int res = context.run();
    if (context.shouldExit())
        return res;  // Propagate test results.

    int client_stuff_return_code = 0;

    return res + client_stuff_return_code;  // Propagate test results.
}