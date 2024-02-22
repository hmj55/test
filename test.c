#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int getPackParam(unsigned char *temp, int tempLen, unsigned char *pack, int *point, int index)
{
    int len = point[index + 1] - point[index] - 1;
    memset(temp, 0, tempLen);
    memcpy(temp, &pack[point[index] + 1], len);
    return len;
}
int doMsg(unsigned char *data, int len)
{
    unsigned char temp[64] = {0};
    int point[20] = {0};
    int pointNums = 0; // 分隔符节点数量

    for (int i = 0; i < len; i++)
    {
        if (data[i] == ',')
            point[pointNums++] = i;
    }

    point[pointNums] = len - 1;

    if (pointNums > 20)
        return -1;

    printf("%d\n", pointNums);
    printf("data :%s\n", data);
    memcpy(temp, &data[1], 3);
    int index = 1;
    int tempLen = getPackParam(temp, sizeof(temp), data, point, index++);
    printf(temp);

    tempLen = getPackParam(temp, sizeof(temp), data, point, index++);
    if (index == pointNums)
        printf(temp);

    return 0;
}

int getPointNum(unsigned char *pack, int packLen)
{
    int point[20] = {0};
    int pointNums = 0;

    for (int i = 0; i < packLen; i++)
    {
        if (pack[i] == ',')
            point[pointNums++] = i;
        if (pack[i] == 0x03)
            point[pointNums++] = i;
    }

    return pointNums;
}

int main(void)
{
    // unsigned char buff[] = {0x02, 'C', 'M', 'D', ',', '1', '2', '2', ',', 0X03};
    // doMsg(buff, sizeof(buff));

    unsigned char pack[] = "SMC,C,1,CHARGE";
    printf("\n%d\n", getPointNum(pack, sizeof(pack)));
    return 0;
}