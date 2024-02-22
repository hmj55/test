#include <stdio.h>
#define GET_LOW_BYTE(num, x) ((num >> x) & 0x01)

int main(void)
{
    int data = 0x40;
    printf("low_byte:%d\n", GET_LOW_BYTE(data, 7));
    return 0;
}