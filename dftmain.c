#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// hash表的链表的节点
typedef struct node
{
    int index;
    int data;          // 存数据
    struct node *next; // 存指针
} HASH;

// 创建hash表,创建一堆对应键值的node数组，重复的键值对点会在键值的节点后面增加。
HASH **create_hash(int nodeSize)
{
    HASH **h = (HASH **)malloc(nodeSize * 8);
    int i = 0;
    for (i = 0; i < nodeSize; i++)
    {
        h[i] = (struct node *)malloc(sizeof(struct node));
        h[i]->next = NULL;
    }

    return h;
}

// 插入数据
/****************
 * 重复的键值对应的元素会添加到键值对应的链表的后面。
 */
int insert_hash_table(HASH **h, int data, int index, int numSize)
{
    int key = (int)fabs(data) % numSize;
    struct node *p = h[key];

    // 头插法插入数据
    struct node *temp;
    temp = (struct node *)malloc(sizeof(struct node));
    temp->data = data;
    temp->index = index;
    temp->next = p->next;
    p->next = temp;

    return 0;
}

// 释放链表节点
int free_hash_table(struct node *head)
{
    // 如果链表后面没有数据，则无需释放
    if (head->next == NULL)
    {
        return 0;
    }

    // 遍历这个链表-头删法释放
    while (head->next != NULL)
    {
        // 定义一个结构体指针变量 来指向这个即将被删除的结构体 以便释放
        struct node *temp = head->next;
        head->next = head->next->next;
        free(temp);
        temp = NULL;
    }
    return 0;
}

// 查找数据
/****************
 * 返回键值链表的第一个节点的值。
 */
int search_hash_table(HASH **h, int data, int numSize)
{
    int key = (int)fabs(data) % numSize; // 数据对质数取余，得到键值
    struct node *p = h[key];             // 找到对应链表

    // 对比要查找的数据
    while (p->next != NULL)
    {
        if (p->next->data == data)
        {
            return p->next->index; // 找到返回index
        }
        p = p->next;
    }
    // 没有找到返回0
    return -1;
}

void print_data(const char *data, int size)
{
    /*     for (int i = 0; i < size; i += 16)
        {
            printf("%02x ", data[i]);
            for (int j = 1; j < 16 && i + j < size; j++)
                printf("%02x ", data[i + j]);

            printf("\n");
        } */

    for (int i = 0; i < size; i++)
    {
        printf("%02x ", data[i]);
        if (i % 15 == 0 && i != 0)
            printf("\n");
    }
}

int *twoSum(int *nums, int numsSize, int target, int *returnSize)
{
    *returnSize = 0;
    int *res = malloc(sizeof(int) * 2);
    res[0] = 0;
    res[1] = 0;
    HASH **h = create_hash(numsSize); // 创建hash表

    for (int i = 0; i < numsSize; i++)
    {
        insert_hash_table(h, nums[i], i, numsSize); // 链表的插入
    }

    for (int i = 0; i < numsSize; i++)
    {
        int searchTemp = target - nums[i];
        int index = search_hash_table(h, searchTemp, numsSize);
        if ((index != -1) && (index != i)) // 防止元素重复
        {
            *returnSize = 2;
            res[0] = i;
            res[1] = index;
            break;
        }
    }

    for (int i = 0; i < numsSize; i++)
    {
        free_hash_table(h[i]);
    }
    free(h);
    return res;
}

struct ListNode
{
    int val;
    struct ListNode *next;
};

// 链表数加法
struct ListNode *addTwoNumbers(struct ListNode *l1, struct ListNode *l2)
{
    struct ListNode *res =
        (struct ListNode *)malloc(sizeof(struct ListNode)); // 用一个哑节点(dummy node)提供迭代的初始参数；
    res->val = 0;
    res->next = NULL; // 初始化操作
    struct ListNode *pre = res;
    // 在后面的操作中会做pre = pre->next;操作，这里是保持res指向的位置不变。
    int carry = 0;            // 记录满十进位。
    while (l1 || l2 || carry) // 只要有一个存在迭代就可以继续；
    {
        struct ListNode *temp = (struct ListNode *)malloc(sizeof(struct ListNode));
        int n1 = l1 ? l1->val : 0;
        int n2 = l2 ? l2->val : 0;
        int sum = n1 + n2 + carry;

        temp->val = sum % 10;
        temp->next = NULL;
        pre->next = temp;
        // 以上三行代码为新节点各域(数据域和指针域)赋值，并放置于前驱节点之后；

        pre = pre->next;
        carry = sum / 10;
        l1 = l1 ? l1->next : NULL;
        l2 = l2 ? l2->next : NULL;
        // 以上四行代码更新四个迭代参数；
    }
    return res->next;
}

//通过使用两个指针i和j，从数组的两端向中间移动，将小于基准值的元素移到左侧，将大于基准值的元素移到右侧，最终将基准值放置在正确的位置上。该函数的作用是对数组进行划分，以便进行快速排序的下一步操作。
int get_standard(int *array, int i, int j)
{
    // 基准数据
    int key = array[i];
    while (i < j)
    {
        // 因为默认基准是从左边开始，所以从右边开始比较
        // 当队尾的元素大于等于基准数据 时,就一直向前挪动 j 指针
        while (i < j && array[j] >= key)
        {
            j--;
        }
        // 当找到比 array[i] 小的时，就把后面的值 array[j] 赋给它
        if (i < j)
        {
            array[i] = array[j];
        }
        // 当队首元素小于等于基准数据 时,就一直向后挪动 i 指针
        while (i < j && array[i] <= key)
        {
            i++;
        }
        // 当找到比 array[j] 大的时，就把前面的值 array[i] 赋给它
        if (i < j)
        {
            array[j] = array[i];
        }
    }
    // 跳出循环时 i 和 j 相等,此时的 i 或 j 就是 key 的正确索引位置
    // 把基准数据赋给正确位置
    array[i] = key;
    return i;
}

void quick_sort(int *array, int low, int high)
{
    // 开始默认基准为 low
    if (low < high)
    {
        // 分段位置下标
        int standard = get_standard(array, low, high);
        // 递归调用排序
        // 左边排序
        quick_sort(array, low, standard - 1);
        // 右边排序
        quick_sort(array, standard + 1, high);
    }
}

#define min(a, b) ((a) < (b) ? (a) : (b))

int getKth(int *nums1, int start1, int end1, int *nums2, int start2, int end2, int K)
{
    int len1 = end1 - start1 + 1;
    int len2 = end2 - start2 + 1;

    // 始终让nums1的长度比nums2的长度小。
    if (len1 > len2)
        return getKth(nums2, start2, end2, nums1, start1, end1, K);

    // nums1数组已近遍历完，目标k数值就在nums2里面。
    if (len1 == 0)
        return nums2[start2 + K - 1];

    // 已经将比目标数值K中位数小的的数值剔除了。
    if (K == 1)
        return min(nums1[start1], nums2[start2]);

    // min(len1, K / 2)：防止排除的数据长度超过nums的大小，导致越界。所以直接指向最后一个。
    // 指向排除掉的数据的后一位
    int i = start1 + min(len1, K / 2) - 1;
    int j = start2 + min(len2, K / 2) - 1;

    // 因为是有序数组，所以小的一个数组元素下的所有元素都小于大的数组元素，就可以将小的元素所在的数组的元素所在位置之下的数组排除，之后再次进行排除，直到其中一个数组遍历完或者k=1。
    if (nums1[i] > nums2[j])
        return getKth(nums1, start1, end1, nums2, j + 1, end2, K - (j - start2 + 1));
    else
        return getKth(nums1, i + 1, end1, nums2, start2, end2, K - (i - start1 + 1));
}

double findMedianSortedArrays(int *nums1, int nums1Size, int *nums2, int nums2Size)
{
    double res = 0;
    int left = (nums1Size + nums2Size + 1) / 2;
    int right = (nums1Size + nums2Size + 2) / 2;
    // 用来解决总数量为奇数偶数的问题，
    // 奇数：（1+2）/2 = 1；（1+1）/2 = 1；
    // 偶数：（2+1）/2 = 1；（2+2）/2 = 2；
    res = 0.5 * (getKth(nums1, 0, nums1Size - 1, nums2, 0, nums2Size - 1, left) +
                 getKth(nums1, 0, nums1Size - 1, nums2, 0, nums2Size - 1, right));
    return res;
}

#define max(a, b) ((a) > (b) ? (a) : (b))
int expandAroundCenter(char *s, int left, int right)
{
    int L = left, R = right;
    while (L >= 0 && R < strlen(s) && s[L] == s[R])
    {
        L--;
        R++;
    }
    return R - L - 1;
}

char *longestPalindrome(char *s)
{
    if (s == NULL || strlen(s) < 1)
        return NULL;

    int start = 0;
    int end = 0;

    // 每次循环选择一个中心，进行左右扩展，判断左右字符是否相等。
    for (int i = 0; i < strlen(s); i++)
    {
        // 因为存在奇数的字符串和偶数的字符串，所以需要从一个字符开始扩展或者从两个祖父之间开始扩展，所有一共有(n+n-1)个中心
        int len1 = expandAroundCenter(s, i, i);
        int len2 = expandAroundCenter(s, i, i + 1);
        int len = max(len1, len2);

        if (len > end - start)
        {
            start = i - (len - 1) / 2;
            end = i + len / 2;
        }

        /***********************
         * 中心移动顺序
         *[a,b,b,a,b,a]==>[(a),b,b,a,b,a]==>[a(,)b,b,a,b,a]
         *计算start和end
         *[{a,b}(,){b,a},b,a]      [a,b,{b},(a),{b},a]
         *start = i - (len - 1) / 2；
         *如果是偶数的字符串的时候i表示中心空位前一个字符，奇数的时候表示中心字符；
         *end = i + len / 2;
         *表示中心位置往后len/2长度的字符位置
         *****************************/
    }
    s[end + 1] = '\0';
    return &s[start];
}

char *convert(char *s, int numRows)
{
    int len = strlen(s);
    if (len <= numRows || numRows < 2)
        return s;
    char **box = (char **)malloc(sizeof(char *) * numRows);
    memset(box, '\0', sizeof(char *) * numRows);
    for (int i = 0; i < numRows; i++)
    {
        box[i] = (char *)malloc(sizeof(char) * len / 2 + 1);
        memset(box[i], 0, sizeof(char) * len / 2 + 1);
    }
    char *result = malloc(sizeof(char) * len + 1);
    memset(result, 0, sizeof(char) * len + 1);

    int j = 0;
    int flag = 1;
    for (int i = 0; i < len; i++)
    {
        strncat(box[j], &s[i], 1);
        printf("index:%d\n", i);
        printf("box[%d]:%s,Len:%d\n", j, box[j], strlen(box[j]));
        j += flag;
        if (j == numRows - 1 || j == 0)
        {
            flag = -flag;
        }
    }

    for (int i = 0; i < numRows; i++)
    {
        printf("%s\n", box[i]);
        strcat(result, box[i]);
        free(box[i]);
    }
    free(box);

    return result;
}

char *convertEx(char *s, int numRows)
{
    int len = strlen(s);
    // 先将不需要排列的数据直接返回。
    if (len < numRows || numRows == 1)
        return s;

    char *convertS = (char *)malloc(sizeof(char) * (len + 1));
    memset(convertS, 0, sizeof(char) * (len + 1));

    int idx = 0;
    for (int i = 1; i <= numRows; ++i)
    {
        if (i == 1 || i == numRows)
        {
            // 第0层和第numRows-1层的下标间距总是step 。step=2*numRows-2
            int tmp = (numRows - 1) * 2;
            printf("tmp:%d\n", tmp);
            for (int j = i - 1; j < len; j += tmp)
            {
                convertS[idx++] = s[j];
            }
        }
        else
        {
            // 中间层的下标间距总是(step-2*行数)，(2*行数)交替。
            int tmp1 = (numRows - i) * 2;
            int tmp2 = (i - 1) * 2;

            printf("tmp1:%d\ttmp2:%d\n", tmp1, tmp2);
            for (int j = i - 1; j < len;)
            {
                convertS[idx++] = s[j];
                j += tmp1;
                if (j < len)
                {
                    convertS[idx++] = s[j];
                    j += tmp2;
                }
            }
        }
        printf("conv:%s\n", convertS);
    }
    return convertS;
}

int reverse(int x)
{
    char temp[24] = {0};
    int num = x;
    int index = 0;
    if (x < 0)
        temp[index++] = '-';

    while (1)
    {
        temp[index++] = fabs(num % 10) + '0';
        num = num / 10;
        if (num == 0)
            break;
    }
    printf("temp:%s\n", temp);
    int res = atoi(temp);
    temp[index - 1] = '\0';
    int check = atoi(temp);
    printf("res:%d\tcheck:%d\n", res, check);
    if (check == res / 10)
        return res;
    else
        return 0;
}

int myAtoi(char *s)
{
    double res = 0;
    int index = 0;
    int flag = 0; // 0正数

    // 跳过空格
    while (s[index] == ' ')
    {
        index++;
    }
    // 判断正负
    if (s[index] == '+' || s[index] == '-')
    {
        if (s[index] == '-')
            flag = 1;

        index++;
    }

    // 计算数据
    while (s[index] >= '0' && s[index] <= '9')
    {
        res += (s[index] - '0');
        if (res >= INT_MAX && flag == 0)
            return INT_MAX;
        if ((0 - res) <= INT_MIN)
            return INT_MIN;
        if (s[index + 1] >= '0' && s[index + 1] <= '9')
            res *= 10;
        index++;
    }

    if (flag == 1)
        return (int)(0 - res);

    return (int)res;
}

int reverse1(int x)
{
    int ret = 0, max = 0x7fffffff, min = 0;
    long rs = 0;
    for (; x; rs = rs * 10 + x % 10, x /= 10)
        ;
    return ret = (((rs > max) || (rs < min)) ? 0 : rs);
}

int isPalindrome(int x)
{
    int ret = x, max = 0x7fffffff, min = 0;
    long rs = 0;
    for (; ret; rs = rs * 10 + ret % 10, ret /= 10)
        ;
    ret = (((rs > max) || (rs < min)) ? 0 : rs);
    return x == ret;
}

int cmp(const void *a, const void *b)
{
    return *(int *)a - *(int *)b;
}

void quickSort(int *a, int l, int r)
{
    qsort(&a[l], r - l + 1, sizeof(int), cmp);
}

void quickSortFloat(float *a, int l, int r)
{
    qsort(&a[l], r - l + 1, sizeof(float), cmp);
}

/*给你一个整数数组 nums 和一个整数 k 。
每一步操作中，你需要从数组中选出和为 k 的两个整数，并将它们移出数组。
返回你可以对数组执行的最大操作数。 */
int maxOperations(int *nums, int numsSize, int k)
{
    int *temp = malloc(sizeof(int) * (numsSize + 1));
    memset(temp, 0, sizeof(int) * (numsSize + 1));
    memcpy(temp, nums, sizeof(int) * numsSize);
    temp[numsSize] = k;
    quickSort(&temp[0], 0, numsSize);
    int index = 0, res = 0;

    for (int i = 0; i < numsSize + 1; i++)
    {
        printf("temp[%d]:%d\n", i, temp[i]);
    }

    while (temp[index] < k)
    {
        index++;
    };
    printf("index:%d\n", index);

    int left = 0;
    int right = index - 1;

    while (left < right)
    {
        if (temp[left] + temp[right] == k)
        {
            left++;
            right--;
            res++;
        }
        else if (temp[left] + temp[right] < k)
            left++;
        else
            right--;
    }
    free(temp);
    return res;
}

int maxArea(int *height, int heightSize)
{
    int left = 0;
    int right = heightSize - 1;
    int len = heightSize - 1;
    int res = 0;

    while (left != right)
    {
        int area = min(height[left], height[right]) * len;
        printf("area:%d\n", area);
        res = res > area ? res : area;

        if (height[left] > height[right])
            right--;
        else
            left++;

        len--;
    }

    return res;
}
const char *thousands[] = {"", "M", "MM", "MMM"};
const char *hundreds[] = {"", "C", "CC", "CCC", "CD", "D", "DC", "DCC", "DCCC", "CM"};
const char *tens[] = {"", "X", "XX", "XXX", "XL", "L", "LX", "LXX", "LXXX", "XC"};
const char *ones[] = {"", "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX"};

char *intToRoman(int num)
{
    char *roman = malloc(sizeof(char) * 16);
    memset(roman, 0, sizeof(char) * 16);
    roman[0] = '\0';

    strcpy(roman + strlen(roman), thousands[num / 1000]);
    strcpy(roman + strlen(roman), hundreds[num % 1000 / 100]);
    strcpy(roman + strlen(roman), tens[num % 100 / 10]);
    strcpy(roman + strlen(roman), ones[num % 10]);
    return roman;
}

const int values[] = {1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1};
const char *symbols[] = {"M", "CM", "D", "CD", "C", "XC", "L", "XL", "X", "IX", "V", "IV", "I"};

char *intToRomanOther(int num)
{
    char *roman = malloc(sizeof(char) * 16);
    memset(roman, 0, sizeof(char) * 16);
    for (int i = 0; i < 13; i++)
    {
        while (num >= values[i])
        {
            num -= values[i];
            strcpy(roman + strlen(roman), symbols[i]);
        }
        if (num == 0)
        {
            break;
        }
    }
    return roman;
}

int change(char s)
{
    switch (s)
    {
    case 'I':
        return 1;
    case 'V':
        return 5;
    case 'X':
        return 10;
    case 'L':
        return 50;
    case 'C':
        return 100;
    case 'D':
        return 500;
    default:
        return 1000;
    }
}

int romanToInt(char *s)
{
    int len = strlen(s);
    int res = 0;
    int last = 0;

    for (int i = len - 1; i >= 0; i--)
    {
        int temp = change(s[i]);
        if (temp >= last)
        {
            res += temp;
            last = temp;
        }
        else
        {
            res -= temp;
        }
    }

    return res;
}

/* ********************
 *用于在数组array中查找一个位置，使其后面的元素满足递增或者递减的要求。
 *可以用来查找波形曲线的底部或者顶部 */
static int startPosition(const float *array, int start, int end, unsigned char isIncrement)
{
#if 1
    int pos = 0;
    int cnt = 0;
    float preValue = array[start];

    for (int i = start + 1; i < end; i++)
    {
        if (isIncrement ? array[i] >= preValue : array[i] <= preValue)
            cnt++;
        else
        {
            cnt = 0;
            pos = i;
        }

        preValue = array[i];
        if (cnt > 5)
            break;
    }
#else

    int pos = start;

    int low = start + 1;
    int high = end - 1;

    while (low <= high)
    {
        int mid = (low + high) / 2;
        if ((isIncrement && array[mid] <= array[mid - 1]) || (!isIncrement && array[mid] >= array[mid - 1]))
        {
            pos = mid;
            low = mid + 1;
        }
        else
        {
            high = mid - 1;
        }
    }
#endif

    return pos;
}

static int upperIndex(const float *array, int start, int end, float target, unsigned char isIncrement)
{
    float score = 9999999.9;
    int index = -1;
    int pos = startPosition(array, start, end, isIncrement);

    /*   for (int i = pos; i < end; i++)
      {
          float tmp = array[i] - target;
          if ((tmp > score) || (tmp < 0) || (fabs(tmp) < 1e-5))
              continue;

          score = tmp;
          index = i;
      } */

    int low = pos;
    int high = end - 1;

    while (low <= high)
    {
        int mid = (low + high) / 2;
        float tmp = array[mid] - target;
        if (tmp <= 0)
        {
            low = mid + 1;
        }
        else
        {
            if (tmp < score)
            {
                score = tmp;
                index = mid;
            }
            high = mid - 1;
        }
    }

    if (index == -1)
        index = array[0] > array[end - 1] ? 0 : end - 1;

    return index;
}

static int lowerIndex(const float *array, int start, int end, float target, unsigned char isIncrement)
{
    float score = -9999999.9;
    int index = -1;
    int pos = startPosition(array, start, end, isIncrement);

    for (int i = pos; i < end; i++)
    {
        float tmp = array[i] - target;

        if ((tmp > 0) || (tmp < score))
            continue;

        score = tmp;
        index = i;
    }

    /* int low = startPosition(array, start, end, isIncrement);
    int high = end - 1;
    int index = -1;

    while (low <= high)
    {
        int mid = (low + high) / 2;
        if (array[mid] <= target)
        {
            index = mid;
            low = mid + 1;
        }
        else
        {
            high = mid - 1;
        }
    } */

    if (index == -1)
        index = array[0] < array[end - 1] ? 0 : end - 1;

    return index;
}

char *longestCommonPrefix(char **strs, int strsSize)
{
    int maxLen = strlen(strs[0]);
    for (int index = 1; index < maxLen; index++)
    {
        for (int i = 1; i < strsSize; i++)
        {
            if (strs[i][index] == '\0')
                return strs[i];

            if (strs[0][index] != strs[i][index])
            {
                strs[0][index] = '\0';
                return strs[0];
            }
        }
    }
    return strs[0];
}

/**
 * Return an array of arrays of size *returnSize.
 * The sizes of the arrays are returned as *returnColumnSizes array.
 * Note: Both returned array and *columnSizes array must be malloced, assume caller calls free().
 */
int **threeSum(int *nums, int numsSize, int *returnSize, int **returnColumnSizes)
{
    int target = 255;
    int base = 100;
    int **res = (int **)malloc(sizeof(int *) * base);
    *returnColumnSizes = (int *)malloc(sizeof(int) * base);
    *returnSize = 0;
    if (numsSize < 3)
        return res;

    // 先从小到大排序
    quickSort(nums, 0, numsSize - 1);

    for (int a = 0; a < numsSize - 2; a++)
    {
        // a位置的值是最小的
        if (nums[a] > 0)
            break;

        // 去重
        if (a > 0 && nums[a] == nums[a - 1])
            continue;

        int b = a + 1;
        int c = numsSize - 1;

        while (b < c)
        {
            int sum = nums[a] + nums[b] + nums[c];
            if (sum < target)
                b++;
            else if (sum > target)
                c--;
            else if (sum == target)
            {
                res[*returnSize] = (int *)malloc(sizeof(int) * 3);
                res[*returnSize][0] = nums[a];
                res[*returnSize][1] = nums[b];
                res[*returnSize][2] = nums[c];

                // 去重
                while (c > b && nums[c] == nums[c - 1])
                    c--;

                // 去重
                while (c < b && nums[b] == nums[b + 1])
                    b++;

                printf("d0:%d\td1:%d\td2:%d\n", nums[a], nums[b], nums[c]);
                (*returnColumnSizes)[*returnSize] = 3;
                (*returnSize)++;
                if ((*returnSize) >= base)
                {
                    base *= 2; // base += 2;时候内存会超出限制
                    res = (int **)realloc(res, sizeof(int *) * base);
                    *returnColumnSizes = (int *)realloc(*returnColumnSizes, sizeof(int) * base);
                }

                b++;
                c--;
            }
        }
    }
    return res;
}

int threeSumClosest(int *nums, int numsSize, int target)
{
    quickSort(nums, 0, numsSize - 1);
    int res = 1e7;

    for (int i = 0; i < numsSize; i++)
    {
        // 过滤掉相同的值
        if (i > 0 && nums[i] == nums[i - 1])
            continue;

        int left = i + 1;
        int right = numsSize - 1;
        while (left < right)
        {
            int sum = nums[i] + nums[left] + nums[right];
            // 已经是最近的值了
            if (sum == target)
                return target;

            // 比较获得小的值
            res = fabs(sum - target) > fabs(res - target) ? res : sum;

            if (sum < target)
            {
                left++; // 增加left对应的增加sum的大小
                while (left < right && nums[left - 1] == nums[left])
                    ++left;
            }
            else
            {
                right--; // 减小right对应的减小sum的大小
                while (left < right && nums[right + 1] == nums[right])
                    --right;
            }
        }
    }

    return res;
}

/* 在回溯算法中，我们需要对当前节点进行操作，并进一步深度遍历它的子节点。当我们遍历完所有的子节点后，需要回溯到当前节点的父节点，在这个过程中需要撤销当前节点的操作。
在这段代码中，pathIndex 表示 path 数组中下一个要填充的字符位置。在每次进入递归时，我们需要对 path
数组中的下一个字符进行填充，并将 pathIndex 增加 1。然后递归进入下一层，等递归返回时需要进行回溯，将 pathIndex 减去
1，撤销对 path 数组的填充*/
// 按键对应字母表
char phoneMap[11][5] = {"\0", "\0", "abc\0", "def\0", "ghi\0", "jkl\0", "mno\0", "pqrs\0", "tuv\0", "wxyz\0"};
void backTrackCombination(char **result, int *resultIndex, char *path, int *pathIndex, char *digits, int startIndex,
                          int len)
{
    /* startIndex == len 是退出递归的条件。当 startIndex 等于 len
     * 时，说明已经组合出一个完整的电话号码，将其加入到结果数组中，并结束递归。在之前的递归过程中，每次深度遍历的都是当前
     * digits 数组中的下一个数字，直到组合出一个完整的电话号码为止。 */
    if (startIndex == len)
    {
        path[*pathIndex] = '\0';
        char *temc = (char *)malloc(sizeof(char) * (len + 1));
        strcpy(temc, path);
        result[*resultIndex] = temc;
        (*resultIndex)++;
    }

    char *target = phoneMap[digits[startIndex] - '0'];
    int tarLen = strlen(target);

    for (int i = 0; i < tarLen; i++)
    {
        path[*pathIndex] = target[i];
        (*pathIndex)++;
        backTrackCombination(result, resultIndex, path, pathIndex, digits, startIndex + 1, len);
        (*pathIndex)--;
    }
    /* pathIndex 表示 path 数组中下一个要填充的字符位置。在每次进入递归时，我们需要对 path
     * 数组中的下一个字符进行填充，并将 pathIndex 增加 1。然后递归进入下一层，等递归返回时需要进行回溯，将 pathIndex
     * 减去 1，撤销对 path 数组的填充。在递归调用中修改了 path 数组中的值，在回溯到当前节点时，我们需要将 path
     * 数组恢复到在进入当前节点时的状态。 */
}

char *result[144];
char path[5];
char **letterCombinations(char *digits, int *returnSize)
{
    *returnSize = 0;
    int digitsSize = strlen(digits);
    if (digitsSize == 0)
        return NULL;

    int pathIndex = 0;

    backTrackCombination(result, returnSize, path, &pathIndex, digits, 0, digitsSize);
    return result;
}

int **fourSum(int *nums, int numsSize, int target, int *returnSize, int **returnColumnSizes)
{
    *returnSize = 0;
    int base = 100;
    int **res = (int **)malloc(sizeof(int *) * base);
    *returnColumnSizes = (int *)malloc(sizeof(int) * base);

    if (numsSize < 4)
        return res;

    quickSort(nums, 0, numsSize - 1);

    for (int a = 0; a < numsSize - 3; a++)
    {
        if ((long)nums[a] + nums[a + 1] + nums[a + 2] + nums[a + 3] > target)
            break;

        if (a > 0 && nums[a] == nums[a - 1])
            continue;

        for (int b = a + 1; b < numsSize - 2; b++)
        {

            if ((long)nums[a] + nums[b] + nums[b + 1] + nums[b + 2] > target)
                break;
            if (b > (a + 1) && nums[b] == nums[b - 1])
                continue;

            int c = b + 1;
            int d = numsSize - 1;

            while (c < d)
            {
                // 防止sum值超过int范围
                long sum = (long)nums[a] + nums[b] + nums[c] + nums[d];
                if (sum < target)
                    c++;
                else if (sum > target)
                    d--;
                else if (sum == target)
                {
                    printf("d0:%d\td1:%d\td2:%d\td3:%d\n", nums[a], nums[b], nums[c], nums[d]);
                    res[*returnSize] = (int *)malloc(sizeof(int) * 4);
                    res[*returnSize][0] = nums[a];
                    res[*returnSize][1] = nums[b];
                    res[*returnSize][2] = nums[c];
                    res[*returnSize][3] = nums[d];

                    while (d > c && nums[d] == nums[d - 1])
                        d--;

                    while (d < c && nums[c] == nums[c + 1])
                        c++;

                    (*returnColumnSizes)[*returnSize] = 4;
                    (*returnSize)++;
                    if ((*returnSize) >= base)
                    {
                        base *= 2; // base += 2;时候内存会超出限制
                        res = (int **)realloc(res, sizeof(int *) * base);
                        *returnColumnSizes = (int *)realloc(*returnColumnSizes, sizeof(int) * base);
                    }

                    c++;
                    d--;
                }
            }
        }
    }

    return res;
}

struct ListNode *removeNthFromEnd(struct ListNode *head, int n)
{
    int nodeSize = 0;
    struct ListNode *dummy = malloc(sizeof(struct ListNode));
    dummy->val = 0;
    dummy->next = head;
    struct ListNode *temp = head;
    while (temp->next)
    {
        ++nodeSize;
        temp = temp->next;
    }
    printf("%d\n", nodeSize);

    temp = dummy;

    struct ListNode *prev = NULL;
    for (int i = 0; i < nodeSize - n + 1; i++)
    {
        temp = temp->next;
    }

    temp->next = temp->next->next;
    struct ListNode *ans = dummy->next;
    free(dummy);
    return ans;
}

struct ListNode *removeNthFromEndEx(struct ListNode *head, int n)
{
    struct ListNode *dummy = malloc(sizeof(struct ListNode));
    dummy->val = 0;
    dummy->next = head;
    struct ListNode *first = head;
    struct ListNode *second = dummy;
    for (int i = 0; i < n; i++)
    {
        first = first->next;
    }

    while (first)
    {
        first = first->next;
        second = second->next;
    }

    second->next = second->next->next;
    struct ListNode *ans = dummy->next;
    free(dummy);
    return ans;
}
/* 满足true条件情况："[]","[]()","{[]}()"
 *false 情况："[","{[}]","]","{[}","{{}(})"
 */
int isValid(char *s)
{
    int len = strlen(s);
    int res = 0;
    char *temp = (char *)malloc(sizeof(char) * len);
    memset(temp, 0, sizeof(char) * len);
    int tempIndex = 0;

    for (int i = 0; i < len; i++)
    {
        if ((s[i] == '(') || (s[i] == '[') || (s[i] == '{'))
        {
            tempIndex++;
            temp[tempIndex] = s[i];
        }
        else if ((s[i] == (temp[tempIndex] + 1)) || (s[i] == (temp[tempIndex] + 2)))
        {
            // '(' + 1 = ')';'{'+2 = '}';'['+2 = ']';
            tempIndex--;
        }
        else
        {
            res = 0;
            break;
        }
    }

    res = tempIndex == 0 ? 1 : 0;

    free(temp);
    return res;
}

/*  合并两个有序链表：
    递归合并节点，当前节点谁小，就让这个较小的节点的next和另一个链表继续递归合并，直到两个链表有一个的nxet不存在了，那就没法分割问题了，只能返回
    */
struct ListNode *mergeTwoLists(struct ListNode *list1, struct ListNode *list2)
{
    if (list1 == NULL)
        return list2;
    if (list2 == NULL)
        return list1;

    if (list1->val < list2->val)
    {
        list1->next = mergeTwoLists(list1->next, list2);
        return list1;
    }
    else
    {
        list2->next = mergeTwoLists(list1, list2->next);
        return list2;
    }
}

// 合并多个有序链表
/*分治法，链表两两合并
 *链表1，链表2，链表3，链表4，链表5，
 *[链表1，链表2，][链表3，链表4，链表5，]
 *[链表1，链表2，][[链表3，][链表4，链表5，]]
 */
struct ListNode *mergeKLists(struct ListNode **lists, int listsSize)
{

    if (listsSize == 0)
        return NULL;
    if (listsSize == 1)
        return lists[0];

    int middle = listsSize / 2;
    struct ListNode *left = mergeKLists(lists, middle);
    struct ListNode *right = mergeKLists(lists + middle, listsSize - middle);

    if (left && right)
        return mergeTwoLists(left, right);
    else if (left)
        return left;
    else
        return right;
    // struct ListNode *dummy = (struct ListNode *)malloc(sizeof(struct ListNode));
    // struct ListNode *tmp = dummy;

    // while (left && right)
    // {
    //     if (left->val < right->val)
    //     {
    //         tmp->next = left;
    //         left = left->next;
    //     }
    //     else
    //     {
    //         tmp->next = right;
    //         right = right->next;
    //     }
    //     tmp = tmp->next;
    // }

    // if (left)
    //     tmp->next = left;
    // else
    //     tmp->next = right;
    // return dummy->next;
}

/* 两两交换链表中的节点
递归的终止条件是链表中没有节点，或者链表中只有一个节点，此时无法进行交换。
*/
struct ListNode *swapPairs(struct ListNode *head)
{
    if (head == NULL || head->next == NULL)
        return head;
    struct ListNode *newHead = head->next;
    head->next = swapPairs(newHead->next);
    newHead->next = head;
    return newHead;
}

/* 反转链表
    递归法：1step:
            L4->NULL;
            L3->L4->L3;
            L3->NULL;
            L4->L3->NULL;
*/
struct ListNode *reverseList(struct ListNode *head)
{
#if 1
    if (head == NULL || head->next == NULL)
        return head;

    struct ListNode *newHead = reverseList(head->next);
    head->next->next = head;
    head->next = NULL;
    return newHead;
#else
    struct ListNode *prev = NULL;
    struct ListNode *curr = head;

    while (curr)
    {
        struct ListNode *next = curr->next;
        curr->next = prev;
        prev = curr;
        curr = next;
    }
    return prev;

#endif
}

// 检查括号完整性
int isParenthesisValid(char *s)
{
    int count = 0;
    while (*s)
    {
        if (*s == '(')
            count++;
        else
        {
            if (count == 0)
                return 0;
            else
                count--;
        }
        s++;
    }
    return 1;
}

void backTrack_Parenthesis(char *path, int *pathIndex, char **result, int *resultIndex, int leftIndex, int rightIndex,
                           int n)
{
    if (leftIndex > rightIndex)
        return;

    if (*pathIndex == n * 2)
    {
        path[*pathIndex] = '\0';
        printf("path:%s\n", path);
        if (isParenthesisValid(path))
        {
            char *temc = (char *)malloc(sizeof(char) * (2 * n + 1));
            memcpy(temc, path, *pathIndex + 1);
            result[*resultIndex] = temc;
            (*resultIndex)++;
        }
        return;
    }

    if (leftIndex)
    {
        path[*pathIndex] = '(';
        (*pathIndex)++;
        backTrack_Parenthesis(path, pathIndex, result, resultIndex, leftIndex - 1, rightIndex, n);
        (*pathIndex)--;
    }
    if (rightIndex)
    {
        path[*pathIndex] = ')';
        (*pathIndex)++;
        backTrack_Parenthesis(path, pathIndex, result, resultIndex, leftIndex, rightIndex - 1, n);
        (*pathIndex)--;
    }
}

void backTrack_ParenthesisEx(char *path, int *pathIndex, char **result, int *resultIndex, int leftIndex, int rightIndex,
                             int n)
{
    if (leftIndex > rightIndex)
        return;

    if (leftIndex == 0 && rightIndex == 0)
    {
        path[*pathIndex] = '\0';
        char *temc = (char *)malloc(sizeof(char) * (2 * n + 1));
        memcpy(temc, path, *pathIndex + 1);
        result[*resultIndex] = temc;
        (*resultIndex)++;
    }
    else
    {
        if (leftIndex)
        {
            path[*pathIndex] = '(';
            (*pathIndex)++;
            backTrack_ParenthesisEx(path, pathIndex, result, resultIndex, leftIndex - 1, rightIndex, n);
            (*pathIndex)--;
        }
        if (rightIndex)
        {
            path[*pathIndex] = ')';
            (*pathIndex)++;
            backTrack_ParenthesisEx(path, pathIndex, result, resultIndex, leftIndex, rightIndex - 1, n);
            (*pathIndex)--;
        }
    }
}

char **generateParenthesis(int n, int *returnSize)
{
    char **result = (char **)malloc(sizeof(char *) * n * 2);
    *returnSize = 0;
    char *path = (char *)malloc(sizeof(char) * (2 * n + 1));
    int pathIndex = 0;
    backTrack_Parenthesis(path, &pathIndex, result, returnSize, n, n, n);
    return result;
}

// 字符串数组反转
void arrReversed(char *buff, int buffLen)
{
    int len = buffLen / 2;
    while (len--)
    {
        if (buff[buffLen - (len + 1)] != buff[len])
        {
            char temp = buff[buffLen - (len + 1)];
            buff[buffLen - (len + 1)] = buff[len];
            buff[len] = temp;
        }
    }
}

/*********************************
 * 将十进制数解析为二进制字符串数组  7==》"111"
 *
 * return :返回的数组由calloc开辟空间，需要对其free();
 ************************/

char *digitalResolution(int num, int *returnSize)
{
    int len = 10;
    char *res = (char *)calloc(1, sizeof(char) * len);
    int temp = num;
    int count = 0;

    while (temp)
    {
        res[count++] = temp % 2 ? '1' : '0';
        temp /= 2;

        if (count >= len)
        {
            len *= 2;
            res = (char *)realloc(res, sizeof(char) * len);
        }
    }

    *returnSize = count;

    arrReversed(res, count);
    return res;
}

/*********************
 * k个一组翻转链表,采用递归的方法
 *
 *                  1->2->3->4->5->NULL
 * recursion 1 :                5->NULL
 * recursion 2 :          4->3->5->NULL
 * recursion 3 :    2->1->4->3->5->NULL
 *
 * */
struct ListNode *reverseKGroup(struct ListNode *head, int k)
{
    struct ListNode *cur = head;
    int count = 0;

    while (cur && count != k)
    {
        cur = cur->next;
        count++;
    }

    // 只有满足k个节点链表才进行反转
    if (count == k)
    {
        // 先遍历到满足k组的最后一个链表组，将最后一个k组链表反转，返回反转后的表头，之后从后往前依次反转k组链表
        cur = reverseKGroup(cur, k);
        while (count)
        {
            struct ListNode *tmp = head->next;
            head->next = cur; // 链接后一个链表组反转后的头节点，或者NULL，或者不满足k个节点链表组的头节点
            cur = head;
            head = tmp;
            count--;
        }
        /***************
         * 1->2->3->nextListHead;k = 3;
         * loop 1:  cur:1-> nextListHead
         *          head:2;count = 2
         * loop 2:  cur:2->1->nextListHead
         *          head:3;count = 1
         * loop 3:  cur:3->2->1->nextListHead
         *          head = nextListHead;count  = 0;
         * */
        head = cur;
    }

    return head;
}

/***********
 * 删除有序数组中的重复项
 */
int removeDuplicates(int *nums, int numsSize)
{
    if (numsSize < 2)
        return 0;

    int fast = 1;
    int slow = 1;

    while (fast < numsSize)
    {
        if (nums[fast] != nums[fast - 1])
        {
            nums[slow] = nums[fast];
            ++slow;
        }
        ++fast;
    }
    return slow;
}

/*移除目标元素 */
int removeElement(int *nums, int numsSize, int val)
{
    if (numsSize == 0)
        return 0;

    int fast = 1;
    int slow = 1;

    while (fast < numsSize)
    {
        if (nums[fast] != val)
        {
            nums[slow] = nums[fast];
            ++slow;
        }
        ++fast;
    }
    return slow;
}

int myStrStr(char *haystack, char *needle)
{
    int hLen = strlen(haystack);
    int nLen = strlen(needle);
    if (hLen < nLen)
        return -1;

    for (int i = 0; i < hLen; i++)
    {
        if (memcmp(haystack + i, needle, nLen) == 0)
            return i;

        if ((hLen - i) < nLen)
            return -1;
    }

    return -1;
}

long lDiv(long a, long b)
{
    if (a < b)
        return 0;

    long count = 1;
    long temp = b;

    while ((temp + temp) <= a)
    {
        count = count + count;
        temp = temp + temp;
    }

    return count + lDiv(a - temp, b);
}

// 除法
int divide(int dividend, int divisor)
{
    if (dividend == 0)
        return 0;
    if (divisor == 1)
        return dividend;
    if (divisor == -1)
    {
        if (dividend > INT_MIN)
            return -dividend;
        return INT_MAX;
    }

    long a = dividend;
    long b = divisor;

    int sign = 1;
    if ((a > 0 && b > 0) || (a < 0 && b < 0))
    {
        sign = -1;
    }

    a = a > 0 ? a : -a;
    b = b > 0 ? b : -b;
    long res = lDiv(a, b);
    if (sign > 0)
        return res > INT_MAX ? INT_MAX : res;
    return -res;
}

void swap(int *a, int *b)
{
    int t = *a;
    *a = *b;
    *b = t;
}

// 排序颠倒
void permutationReverse(int *nums, int left, int right)
{
    while (left < right)
    {
        swap(nums + left, nums + right);
        left++;
        right--;
    }
}

/**************8
 * 31，下一个排列
 * example:
 * 4,5,2,6,3,1              ==>start
 * step 1:  4,5,(2),6,3,1   ==>较小数2,从后往前的升序结束后的第一个元素
 * step 2:  4,5,(2),5,(3),1 ==>较大数3，升序元素中比较小数2大的元素中最小的元素
 * step 3:  4,5,(3),5,(2),1 ==>swep较大数和较小数
 * step 4:  4,5,(3),(5,2,1) ==>颠倒交换后的较大数后面元素的排序
 * step 5:  4,5,(3),(1,2,5) ==>颠倒交换后的较大数后面元素的排序
 * 4,5,3,1,2,5              ==>end
 *  */
void nextPermutation(int *nums, int numsSize)
{
    int i = numsSize - 2;
    // 寻找一个左边的较小数，较小数后面的数是降序排列
    while (i >= 0 && nums[i] >= nums[i + 1])
    {
        i--;
    }

    printf("i:%d\n", i);

    if (i >= 0)
    {
        int j = numsSize - 1;
        // 寻找右边的较大数，较大数尽可能小，比较小数后面元素中比较小数大的其他元素都要小
        while (j >= 0 && nums[i] >= nums[j])
        {
            j--;
        }
        // 交换较大数和较小数的位置
        swap(nums + i, nums + j);
    }

    // 将交换后的较大数后面的数排序颠倒，因为交换为之后还是满足降序，直接颠倒顺序就可以得到一个最小的排列
    permutationReverse(nums, i + 1, numsSize - 1);
}
/*******
 * 搜索查找位置
 */
int searchInsert(int *nums, int numsSize, int target)
{
    int left = 0;
    int right = numsSize - 1;

    while (left <= right)
    {
        int mid = ((right - left) / 2) + left;
        if (target <= nums[mid])
        {
            right = mid - 1;
        }
        else
        {
            left = mid + 1;
        }
    }
    return left;
}

int lengthOfLastWord1(char *s)
{
    int len = strlen(s);
    int count = 0;

    for (int i = len - 1; i >= 0; i--)
    {
        if (s[i] != ' ')
            count++;
        if (s[i] == ' ' && count > 0)
            break;
    }

    return count;
}

int lengthOfLastWord(char *s)
{
    int len = strlen(s);
    int index = len - 1;
    while (s[index] == ' ')
    {
        index--;
    }

    int ret = 0;
    while (index >= 0 && s[index] != ' ')
    {
        ret++;
        index--;
    }
    return ret;
}

int *plusOne(int *digits, int digitsSize, int *returnSize)
{
    for (int i = digitsSize - 1; i >= 0; i--)
    {
        // 遇到非9的数加1结束循环
        if (digits[i] != 9)
        {
            digits[i]++;
            break;
        }
        // 遇到9加1变0
        digits[i] = 0;
    }

    // 如果最高位为0，表示数为全9组成的数。需要扩大空间
    if (digits[0] == 0)
    {
        int *temp;
        int size = digitsSize + 1;
        temp = (int *)malloc(size * sizeof(int));
        memset(temp, 0, sizeof(int) * size);
        temp[0] = 1;
        *returnSize = size;
        return temp;
    }

    *returnSize = digitsSize;
    return digits;
}

// 二进制求和
char *addBinary(char *a, char *b)
{
    int lenA = strlen(a);
    int lenB = strlen(b);
    int maxLen = lenA > lenB ? lenA : lenB;
    char *res = (char *)malloc(sizeof(char) * (maxLen + 2));
    res[maxLen + 1] = '\0';
    char *pointA = a + lenA - 1;
    char *pointB = b + lenB - 1;
    int carry = 0;
    int resIndex = maxLen;

    while ((pointA >= a) || (pointB >= b) || carry)
    {
        int sum = 0;
        int numA = (pointA >= a) ? (pointA[0] == '0' ? 0 : 1) : 0;
        int numB = (pointB >= b) ? (pointB[0] = '0' ? 0 : 1) : 0;
        sum = numA + numB + carry;
        res[resIndex] = '0' + sum % 2;
        carry = sum / 2;
        --resIndex;
        --pointA;
        --pointB;
    }

    return res[0] == '1' ? res : res + 1;
}

void xBitData28BitData(unsigned char *wData, int dataLen, int xBit)
{
    unsigned int wBuff[1024] = {0};
    int len = xBit / 8 + 1;
    int wLen = 0;

    for (int i = 0; i < dataLen;)
    {
        for (int j = 0; j < len; j++)
        {
            wBuff[wLen] |= wData[i] << (8 * (len - j - 1));
            i++;
        }
        wLen++;
    }

    printf("%d\n", wLen);

    for (int i = 0; i < wLen; i++)
    {
        printf("0x%02x\t", wBuff[i]);
    }
    printf("\n");
}

void bitData2xBitData(unsigned int *rData, int dataLen, int xBit)
{
    unsigned char rBuff[1024] = {0};
    int index = 0;
    int len = 0;

    if (xBit <= 8)
        len = 1;
    else if (xBit <= 16)
        len = 2;
    else if (xBit <= 24)
        len = 3;
    else if (xBit <= 32)
        len = 4;

    for (int i = 0; i < dataLen; i++)
    {
        for (int j = 0; j < len; j++)
        {
            rBuff[index++] = rData[i] >> (8 * (len - 1 - j));
        }
    }

    printf("%d\n", index);

    for (int i = 0; i < index; i++)
    {
        printf("0x%02x\t", rBuff[i]);
    }
    printf("\n");
}

/*************
 * 搜索旋转排序数组
 */
int search(int *nums, int numsSize, int target)
{
    if (numsSize == 1)
    {
        if (nums[0] == target)
        {
            return 0;
        }
        return -1;
    }

    int leftIndex = 0;
    int rightIndex = numsSize - 1;
    int minddleIndex = 0;

    while (leftIndex < rightIndex)
    {
        minddleIndex = (rightIndex - leftIndex) / 2 + leftIndex;

        if (target == nums[minddleIndex])
            return minddleIndex;

        if (target == nums[leftIndex])
            return leftIndex;

        if (target == nums[rightIndex])
            return rightIndex;

        if (nums[leftIndex] < nums[minddleIndex])
        {
            if (nums[leftIndex] < target && target < nums[minddleIndex])
                rightIndex = minddleIndex - 1;
            else
                leftIndex = minddleIndex + 1;
        }
        else
        {

            if (nums[minddleIndex] < target && target < nums[rightIndex])
                leftIndex = minddleIndex + 1;
            else
                rightIndex = minddleIndex - 1;
        }
    }

    return -1;
}

int binarySearch(int *nums, int numSize, int target, int lower)
{
    int left = 0;
    int right = numSize - 1;
    int ans = numSize;

    while (left <= right)
    {
        int mid = (left + right) / 2;
        if (nums[mid] > target || (lower && nums[mid] >= target))
        {
            right = mid - 1;
            ans = mid;
        }
        else
        {
            left = mid + 1;
        }
    }
    return ans;
}
/**
 * Note: The returned array must be malloced, assume caller calls free().
 * 在排序数组中查找元素的第一个和最后一个位置
 */
int *searchRange(int *nums, int numsSize, int target, int *returnSize)
{
    int leftIndex = binarySearch(nums, numsSize, target, 1);
    int rightIndex = binarySearch(nums, numsSize, target, 0) - 1;
    int *ret = malloc(sizeof(int) * 2);
    *returnSize = 2;
    ret[0] = -1;
    ret[1] = -1;
    if (leftIndex <= rightIndex && rightIndex < numsSize && nums[leftIndex] == target && nums[rightIndex] == target)
    {
        ret[0] = leftIndex;
        ret[1] = rightIndex;
    }
    return ret;
}
/********
 * 链表判断是否有环
 * 快慢指针呢个，如果快指针和慢指针相遇，说明存在环
 */
bool hasCycle(struct ListNode *head)
{

    if (head == NULL || head->next == NULL)
        return false;

    struct ListNode *slow = head;
    struct ListNode *fast = head->next;

    while (slow != fast)
    {
        if (fast == NULL || fast->next == NULL)
            return false;
        slow = slow->next;
        fast = fast->next->next;
    }
    return true;
}

/********
 * 判断链表是否存在环，存在的话返回入环的第一个节点
 * fast = 2* slow前提下，fast在slow入圈第一圈即可相遇。
 * 1，快指针走的是慢指针的两遍
 * 2，慢指针走过的路，快指针走过一遍
 * 3，快指针走过的剩余路程，也就是和慢指针走过的全部路程相等。
 * 4，抛去快指针追赶慢指针的半圈，剩余路程即为所求入环距离。
 */
struct ListNode *detectCycle(struct ListNode *head)
{
    struct ListNode *slow = head;
    struct ListNode *fast = head;

    while (fast != NULL)
    {
        slow = slow->next;
        if (fast->next == NULL)
            return NULL;

        fast = fast->next->next;

        if (fast == slow)
        {
            struct ListNode *ptr = head;
            while (ptr != slow)
            {
                ptr = ptr->next;
                slow = slow->next;
            }

            return ptr;
        }
    }
    return NULL;
}

/**********
 * 最大子数组和
 * 动态规划方法，顺序增加到顺序数组所能加到的最大值，将他和之前的值作比较，最后输出最大值。
 * 如果前面的累加起来还没有本身大，就丢弃前面的累加数据，最后输出出现过的最大值。
 */
int maxSubArray(int *nums, int numsSize)
{
    int pre = 0;
    int maxAns = nums[0];

    for (int i = 0; i < numsSize; i++)
    {
        pre = fmax(pre + nums[i], nums[i]);
        maxAns = fmax(maxAns, pre);
    }
    return maxAns;
}

/********
 * 字符串相加
 */
char *addStrings(char *num1, char *num2)
{
    int len1 = strlen(num1) - 1;
    int len2 = strlen(num2) - 1;
    int add = 0;
    char *ans = (char *)malloc(sizeof(char) * (fmax(len1, len2) + 3));
    int len = 0;
    while (len1 >= 0 || len2 >= 0 || add != 0)
    {
        int x = (len1 >= 0) ? (num1[len1] - '0') : 0;
        int y = (len2 >= 0) ? (num2[len2] - '0') : 0;
        int result = x + y + add;
        ans[len++] = '0' + result % 10;
        add = result / 10;
        len1--;
        len2--;
    }

    // 数组反转
    for (int i = 0; 2 * i < len; i++)
    {
        int t = ans[i];
        ans[i] = ans[len - i - 1];
        ans[len - i - 1] = t;
    }
    ans[len++] = 0;
    return ans;
}
/*****
 * 反转字符串
 */
void reverseString(char *s, int sSize)
{
    for (int i = 0; 2 * i < sSize; i++)
    {
        char temp = s[i];
        s[i] = s[sSize - i - 1];
        s[sSize - i - 1] = temp;
    }
}
/************
 * 爬楼梯，动态规划方法
 * f(x) = f(x-1)+f(x-2)
 * 每次只能爬1个台阶或者两个台阶，爬到第n个台阶的方案数满足爬到第n-1个台阶的方案数和爬到第n-2个台阶的方案数之和。
 */
int climbStairs(int n)
{
#if 0
    double fx = pow(((1 + sqrt(5)) / 2), n + 1) / sqrt(5) - pow(((1 - sqrt(5)) / 2), n + 1) / sqrt(5);
    return (int)round(fx);
#endif
    int a = 0;
    int b = 0;
    int res = 1;
    for (int i = 1; i <= n; i++)
    {
        a = b;
        b = res;
        res = a + b;
    }
    return res;
}
/***********
 * 斐波那契数列
 * F(0) = 0;F(1) = 1
 * F(N) = F(N-1) + F(N-2) N > 1
 * 进阶可以用数学推导递推公式特征方程来计算。
 */
int fib(int n)
{
    if (n <= 1)
        return n;
    int a = 0;
    int b = 0;
    int res = 1;
    for (int i = 2; i <= n; i++)
    {
        a = b;
        b = res;
        res = a + b;
    }
    return res;
}

/****
 * 移动零，将数组中的0放到数组后面不影响数组非零字符顺序。
 * 类似快排思想，以0为中间点，把不等于0的放到中间点左边，把等于0的放到中间点右边
 */
void moveZeroes(int *nums, int numsSize)
{
    if (!nums)
        return;
    int j = 0;
    for (int i = 0; i < numsSize; i++)
    {
        if (nums[i] != 0)
        {
            int temp = nums[i];
            nums[i] = nums[j];
            nums[j++] = temp;
        }
    }
}
/*********
 * 移除元素，删除数组中指定元素
 * 快慢指针做法，快指针：寻找不等于目标元素的指针
 *              慢指针：指向快指针找寻到的元素覆盖的位置
 */
int removeElementEX(int *nums, int numsSize, int val)
{
#if 0
    int slow = 0;
    for (int fast = 0; fast < numsSize; fast++)
    {
        if (nums[fast] != val)
            nums[slow++] = nums[fast];
    }
    return slow;
#endif

    int left = 0;
    int right = numsSize;

    while (left < right)
    {
        if (nums[left] = val)
        {
            nums[left] = nums[right - 1];
            right--;
        }
        else
        {
            left++;
        }
    }
    return left;
}

/*************
 * 将数组和减半的最少操作次数
 * 数组其中一个元素减半，之后求和，如果数组和比初始数组和的一半要大，继续对操作过后的数组选择一个元素减半进行求和比较操作，直到数组和小于原始数组和的一半，输出最快需要操作几次。
 * 方法：每次对最大的一个元素做减半操作是最快的。
 */
int halveArray(int *nums, int numsSize)
{
    float *buff = (float *)malloc(sizeof(float) * numsSize);
    int oldSum = 0;
    int res = 0;
    for (int i = 0; i < numsSize; i++)
    {
        oldSum += nums[i];
        buff[i] = nums[i];
    }
    float newSum = oldSum;
    quickSortFloat(buff, 0, numsSize - 1);
    while (newSum > oldSum / 2)
    {
        buff[numsSize - 1] /= 2;
        newSum -= buff[numsSize - 1];
        for (int i = numsSize - 1; i > 0; i--)
        {
            if (buff[i] >= buff[i - 1])
                break;
            float temp = buff[i];
            buff[i] = buff[i - 1];
            buff[i - 1] = temp;
        }
        res++;
    }
    free(buff);

    return res;
}
/**********
 * 长度最小的子数组
 * 给定一个含有 n 个正整数的数组和一个正整数 target 。
 * 找出该数组中满足其和 ≥ target 的长度最小的 连续子数组 [numsl, numsl+1, ..., numsr-1, numsr]
 * ，并返回其长度。如果不存在符合条件的子数组，返回 0 。
 *
 */
int minSubArrayLen(int target, int *nums, int numsSize)
{

    if (numsSize == 0)
        return 0;
#if 0

    int ans = INT_MAX;
    for (int i = 0; i < numsSize; i++)
    {
        int sum = 0;
        for (int j = i; j < nums; j++)
        {
            sum + nums[j];
            if (sum >= target)
            {
                ans = fmin(ans, j - i + 1);
                break;
            }
        }
    }

#endif

    // 滑动窗口
    int ans = INT_MAX;
    int start = 0;
    int end = 0;
    int sum = 0;

    while ((end < numsSize))
    {
        sum += nums[end];
        while (sum >= target)
        {
            ans = fmin(ans, end - start + 1);
            sum -= nums[start];
            start++;
        }
        end++;
    }

    return ans == INT_MAX ? 0 : ans;
}

/********
 * 二分查找
 */
int searchHalf(int *nums, int numsSize, int target)
{
    int left = 0;
    int right = numsSize;
    int mid = 0;

    while (left < right)
    {
        mid = (right - left) / 2 + left;
        if (nums[mid] == target)
            return mid;
        else if (nums[mid] > target)
            right = mid;
        else
            left = mid + 1;
    }

    return -1;
}

/**********
 * 合并两个有序数组
 * 给你两个按 非递减顺序 排列的整数数组 nums1 和 nums2，另有两个整数 m 和 n ，分别表示 nums1 和 nums2 中的元素数目。
 * 请你 合并 nums2 到 nums1 中，使合并后的数组同样按 非递减顺序 排列。
 * 注意：最终，合并后数组不应由函数返回，而是存储在数组 nums1 中。为了应对这种情况，nums1 的初始长度为 m + n，其中前 m
 * 个元素表示应合并的元素，后 n 个元素为 0 ，应忽略。nums2 的长度为 n 。
 */
void merge(int *nums1, int nums1Size, int m, int *nums2, int nums2Size, int n)
{
    int i = m + n - 1;
    m--;
    n--;

    while (n >= 0)
    {
        while ((m >= 0 && nums1[m] > nums2[n]))
        {
            swap(&nums1[i--], &nums1[m--]);
        }
        swap(&nums1[i--], &nums2[n--]);
    }
}
/******
 * 跳跃游戏
 * 给定一个非负整数数组 nums ，你最初位于数组的 第一个下标 。
 * 数组中的每个元素代表你在该位置可以跳跃的最大长度。
 * 判断你是否能够到达最后一个下标。
 */
bool canJump(int *nums, int numsSize)
{
    int cover = 0;
#if 0
    for (int i = 0; i <= cover; i++)
    {
        cover = fmax(nums[i] + i, cover);
        if (cover >= (numsSize - 1))
            return true;
    }
    
    return false;
#endif

    for (int i = 0; i < numsSize; i++)
    {
        // 表示不可能跳到坐标i的位置
        if (i > cover)
            return false;

        // 记录下每次能跳到的最远距离
        cover = fmax(cover, i + nums[i]);

        if (cover >= (numsSize - 1))
            return true;
    }
    return true;
}

/**************
 * 买卖股票的最佳时机
 * 给定一个数组 prices ，它的第 i 个元素 prices[i] 表示一支给定股票第 i 天的价格。
 * 你只能选择 某一天 买入这只股票，并选择在 未来的某一个不同的日子 卖出该股票。设计一个算法来计算你所能获取的最大利润。
 * 返回你可以从这笔交易中获取的最大利润。如果你不能获取任何利润，返回 0 。
 * **/
int maxProfit(int *prices, int pricesSize)
{
#if 0
    int profit = 0;
    for (int i = 0; i < pricesSize; i++)
    {
        for (int j = i + 1; j < pricesSize; j++)
        {
            profit = fmax(prices[j] - prices[i], profit);
        }
    }
    profit = (profit < 0) ? 0 : profit;
    return profit;
#endif
    int maxProfit = 0;

#if 0
    int minPrice = prices[0];
    for (int i = 0; i < pricesSize; i++)
    {
        maxProfit = max(maxProfit, prices[i] - minPrice);
        minPrice = min(prices[i], minPrice);
    }

#endif

    int low = 0;
    int fast = 1;
    while (fast < pricesSize)
    {
        if (prices[low] < prices[fast])
        {
            // 查找最大利润
            if (maxProfit < (prices[fast] - prices[low]))
                maxProfit = prices[fast] - prices[low];

            fast++;
        }
        else
        {
            // 当low元素大于fast元素时候，后面出现的元素减去fast元素的利润肯定比减去low元素的利润大，所以利润计算直接从fast元素开始
            low = fast;
            fast += 1;
        }
    }

    return maxProfit;
}

/***********
 * x的平方根
 */
int mySqrt(int x)
{
    int l = 0;
    int r = x;
    int ans = -1;
    while (l <= r)
    {
        int mid = l + (r - l) / 2;
        if ((long long)mid * mid <= x)
        {
            ans = mid;
            l = mid + 1;
        }
        else
        {
            r = mid - 1;
        }
    }
    return ans;
}

/**************
 * 柠檬水找零
 */
bool lemonadeChange(int *bills, int billsSize)
{
    int five = 0;
    int ten = 0;

    for (int i = 0; i < billsSize; i++)
    {
        if (bills[i] == 5)
            five++;
        else if (bills[i] == 10)
        {
            if (five == 0)
                return false;
            five--;
            ten++;
        }
        else
        {
            if (five > 0 && ten > 0)
            {
                five--;
                ten--;
            }
            else if (five >= 3)
            {
                five -= 3;
            }
            else
            {
                return false;
            }
        }
    }
    return true;
}

/*******
 * 环形子数组的最大和
 *  情况1：构成最大子数组和的子数组为 nums[i:j]，包括 nums[i] 到nums[j?1] 共j?i 个元素，其中 0≤i<j≤n
 *  情况2：构成最大子数组和的子数组为 nums[0:i] 和nums[j:n]，其中 0<i<j<n
 */
int maxSubarraySumCircular(int *nums, int numsSize)
{
#if 0
    int n = numsSize;
    int leftMax[n];
    leftMax[0] = nums[0];
    int leftSum = nums[0];
    int pre = nums[0];
    int res = nums[0];

    for (int i = 1; i < n; i++)
    {
        
        pre = fmax(pre + nums[i], nums[i]);
        res = fmax(res, pre);
        leftSum += nums[i];
        // 右端点坐标范围在[0,i]的最大前缀和可以用leftMax[i]
        leftMax[i] = fmax(leftMax[i - 1], leftSum);
    }

    int rightSum = 0;
    for (int i = n - 1; i > 0; i--)
    {
        
        rightSum += nums[i];
        res = fmax(res, rightSum + leftMax[i - 1]);
    }

    return res;
#endif

#if 0
//环形可以推广看成一个双倍长度的数组的最大和，
    int n = numsSize;
    int deque[n * 2][2];
    int pre = nums[0];
    int res = nums[0];
    int head = 0;
    int tail = 0;
    deque[tail][0] = 0;
    deque[tail][1] = pre;
    tail++;

    for (int i = 1; i < 2 * n; i++)
    {
        while (head != tail && deque[head][0] < i - n)
        {
            head++;
        }

        pre += nums[i % n];
        res = fmax(res, pre - deque[head][1]);

        while (head != tail && deque[tail - 1][1] >= pre)
        {
            tail--;
        }

        deque[tail][0] = i;
        deque[tail][1] = pre;
        tail++;
    }
    return res;

#endif
    // 环形数组的最大子数组和 = 普通数组和 - 普通数组的最小子数组和
    int preMax = nums[0];
    int maxRes = nums[0]; // 普通数组的最大子数组和
    int preMin = nums[0];
    int minRes = nums[0]; // 普通数组的最小子数组和
    int sum = nums[0];
    for (int i = 1; i < numsSize; i++)
    {
        // 计算普通数组的最大子数组和
        preMax = fmax(preMax + nums[i], nums[i]);
        maxRes = fmax(maxRes, preMax);
        // 计算普通数组的最小子数组和
        preMin = fmin(preMin + nums[i], nums[i]);
        minRes = fmin(minRes, preMin);
        // 计算普通数组的和
        sum += nums[i];
    }

    // maxRes<0,表示数组中不包含大于等于0的元素，minRes将包括数组中的所有元素，实际取到的子数组为空，所以最大子数组和就是maxRes本身，最大的一个负数元素。
    if (maxRes < 0)
        return maxRes;
    else
        return fmax(maxRes, sum - minRes);
}
/**********
 * 任意子数组和的绝对值的最大值
 */
int maxAbsoluteSum(int *nums, int numsSize)
{
    int positiveMax = 0;
    int negativeMin = 0;
    int positiveSum = 0;
    int negativeSum = 0;

    for (int i = 0; i < numsSize; i++)
    {
        positiveSum += nums[i];
        positiveMax = fmax(positiveMax, positiveSum);
        positiveSum = fmax(0, positiveSum);
        negativeSum += nums[i];
        negativeMin = fmin(negativeMin, negativeSum);
        negativeSum = fmin(0, negativeSum);
    }

    return fmax(positiveMax, -negativeMin);
}

int main(int argc, const char *argv[])
{
    // char *data = "ABCDEFGHIJK";
    // char *data1 = "abc";
    // char *resStr = intToRomanOther(123);
    // printf("len1:%d\tlen2:%d\n", strlen(data), strlen(resStr));
    // // printf("\n%s\n", resStr);
    // int resNum = romanToInt("III");
    // printf("%d\n", resNum);

    // float array[] = {-1, -2, 1, 2, 3, 4, 5, 4, 5, 6};
    // int arrsize = sizeof(array) / sizeof(array[0]);
    // int index = startPosition(array, 0, arrsize, 1);
    // printf("array[%d]:%.2f\n", index, array[index]);

    // int target = 2;
    // index = upperIndex(array, 0, arrsize, target, 1);
    // printf("upp array[%d]:%.2f\n", index, array[index]);

    // index = lowerIndex(array, 0, arrsize, target, 1);
    // printf("low array[%d]:%.2f\n", index, array[index]);

    // char data[][3] = {"ab", "a"};
    // char *strs = data[0];
    // printf("str:%s\n", longestCommonPrefix(&strs, 2));
    // int arr[] = {1, -1, -1, 0};
    // int returnSize = 0;
    // int *columnSize = NULL;
    // printf("%s\n", "start");
    // threeSum(arr, sizeof(arr) / sizeof(arr[0]), &returnSize, &columnSize);

    // int arr[] = {1, -2, -5, -4, -3, 3, 3, 5};
    // fourSum(arr, sizeof(arr) / sizeof(arr[0]), -11, &returnSize, &columnSize);
    // printf("returnSize:%d\n", returnSize);

    // char data[] = "(";
    // int res = isValid(data);
    // int res = 0;

    // generateParenthesis(4, &res);
    // printf("res:%d\n", res);

    // char data[] = "124";
    // arrReversed(data, strlen(data));
    // printf("%s\n", data);

    // int size = 0;
    // char *buff = digitalResolution(4, &size);
    // printf("size:%d\n", size);
    // printf("%d\n", divide(10, 3));

    // unsigned char data[6] = {0x01, 0x39, 0x02, 0x34, 0x11, 0x01};
    // xBitData28BitData(data, 6, 9);

    // unsigned int rData[6] = {0x0139, 0x2239, 0x0203};
    // bitData2xBitData(rData, 3, 9);

    // int data[5] = {5, 1, -2, 4, -1};
    // int rs = maxSubarraySumCircular(data, 5);
    // printf("%d\n", rs);

    print_data("123456789012345678901234567890", 30);

    return 0;
}