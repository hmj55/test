#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// hash�������Ľڵ�
typedef struct node
{
    int index;
    int data;          // ������
    struct node *next; // ��ָ��
} HASH;

// ����hash��,����һ�Ѷ�Ӧ��ֵ��node���飬�ظ��ļ�ֵ�Ե���ڼ�ֵ�Ľڵ�������ӡ�
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

// ��������
/****************
 * �ظ��ļ�ֵ��Ӧ��Ԫ�ػ���ӵ���ֵ��Ӧ������ĺ��档
 */
int insert_hash_table(HASH **h, int data, int index, int numSize)
{
    int key = (int)fabs(data) % numSize;
    struct node *p = h[key];

    // ͷ�巨��������
    struct node *temp;
    temp = (struct node *)malloc(sizeof(struct node));
    temp->data = data;
    temp->index = index;
    temp->next = p->next;
    p->next = temp;

    return 0;
}

// �ͷ�����ڵ�
int free_hash_table(struct node *head)
{
    // ����������û�����ݣ��������ͷ�
    if (head->next == NULL)
    {
        return 0;
    }

    // �����������-ͷɾ���ͷ�
    while (head->next != NULL)
    {
        // ����һ���ṹ��ָ����� ��ָ�����������ɾ���Ľṹ�� �Ա��ͷ�
        struct node *temp = head->next;
        head->next = head->next->next;
        free(temp);
        temp = NULL;
    }
    return 0;
}

// ��������
/****************
 * ���ؼ�ֵ����ĵ�һ���ڵ��ֵ��
 */
int search_hash_table(HASH **h, int data, int numSize)
{
    int key = (int)fabs(data) % numSize; // ���ݶ�����ȡ�࣬�õ���ֵ
    struct node *p = h[key];             // �ҵ���Ӧ����

    // �Ա�Ҫ���ҵ�����
    while (p->next != NULL)
    {
        if (p->next->data == data)
        {
            return p->next->index; // �ҵ�����index
        }
        p = p->next;
    }
    // û���ҵ�����0
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
    HASH **h = create_hash(numsSize); // ����hash��

    for (int i = 0; i < numsSize; i++)
    {
        insert_hash_table(h, nums[i], i, numsSize); // ����Ĳ���
    }

    for (int i = 0; i < numsSize; i++)
    {
        int searchTemp = target - nums[i];
        int index = search_hash_table(h, searchTemp, numsSize);
        if ((index != -1) && (index != i)) // ��ֹԪ���ظ�
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

// �������ӷ�
struct ListNode *addTwoNumbers(struct ListNode *l1, struct ListNode *l2)
{
    struct ListNode *res =
        (struct ListNode *)malloc(sizeof(struct ListNode)); // ��һ���ƽڵ�(dummy node)�ṩ�����ĳ�ʼ������
    res->val = 0;
    res->next = NULL; // ��ʼ������
    struct ListNode *pre = res;
    // �ں���Ĳ����л���pre = pre->next;�����������Ǳ���resָ���λ�ò��䡣
    int carry = 0;            // ��¼��ʮ��λ��
    while (l1 || l2 || carry) // ֻҪ��һ�����ڵ����Ϳ��Լ�����
    {
        struct ListNode *temp = (struct ListNode *)malloc(sizeof(struct ListNode));
        int n1 = l1 ? l1->val : 0;
        int n2 = l2 ? l2->val : 0;
        int sum = n1 + n2 + carry;

        temp->val = sum % 10;
        temp->next = NULL;
        pre->next = temp;
        // �������д���Ϊ�½ڵ����(�������ָ����)��ֵ����������ǰ���ڵ�֮��

        pre = pre->next;
        carry = sum / 10;
        l1 = l1 ? l1->next : NULL;
        l2 = l2 ? l2->next : NULL;
        // �������д�������ĸ�����������
    }
    return res->next;
}

//ͨ��ʹ������ָ��i��j����������������м��ƶ�����С�ڻ�׼ֵ��Ԫ���Ƶ���࣬�����ڻ�׼ֵ��Ԫ���Ƶ��Ҳ࣬���ս���׼ֵ��������ȷ��λ���ϡ��ú����������Ƕ�������л��֣��Ա���п����������һ��������
int get_standard(int *array, int i, int j)
{
    // ��׼����
    int key = array[i];
    while (i < j)
    {
        // ��ΪĬ�ϻ�׼�Ǵ���߿�ʼ�����Դ��ұ߿�ʼ�Ƚ�
        // ����β��Ԫ�ش��ڵ��ڻ�׼���� ʱ,��һֱ��ǰŲ�� j ָ��
        while (i < j && array[j] >= key)
        {
            j--;
        }
        // ���ҵ��� array[i] С��ʱ���ͰѺ����ֵ array[j] ������
        if (i < j)
        {
            array[i] = array[j];
        }
        // ������Ԫ��С�ڵ��ڻ�׼���� ʱ,��һֱ���Ų�� i ָ��
        while (i < j && array[i] <= key)
        {
            i++;
        }
        // ���ҵ��� array[j] ���ʱ���Ͱ�ǰ���ֵ array[i] ������
        if (i < j)
        {
            array[j] = array[i];
        }
    }
    // ����ѭ��ʱ i �� j ���,��ʱ�� i �� j ���� key ����ȷ����λ��
    // �ѻ�׼���ݸ�����ȷλ��
    array[i] = key;
    return i;
}

void quick_sort(int *array, int low, int high)
{
    // ��ʼĬ�ϻ�׼Ϊ low
    if (low < high)
    {
        // �ֶ�λ���±�
        int standard = get_standard(array, low, high);
        // �ݹ��������
        // �������
        quick_sort(array, low, standard - 1);
        // �ұ�����
        quick_sort(array, standard + 1, high);
    }
}

#define min(a, b) ((a) < (b) ? (a) : (b))

int getKth(int *nums1, int start1, int end1, int *nums2, int start2, int end2, int K)
{
    int len1 = end1 - start1 + 1;
    int len2 = end2 - start2 + 1;

    // ʼ����nums1�ĳ��ȱ�nums2�ĳ���С��
    if (len1 > len2)
        return getKth(nums2, start2, end2, nums1, start1, end1, K);

    // nums1�����ѽ������꣬Ŀ��k��ֵ����nums2���档
    if (len1 == 0)
        return nums2[start2 + K - 1];

    // �Ѿ�����Ŀ����ֵK��λ��С�ĵ���ֵ�޳��ˡ�
    if (K == 1)
        return min(nums1[start1], nums2[start2]);

    // min(len1, K / 2)����ֹ�ų������ݳ��ȳ���nums�Ĵ�С������Խ�硣����ֱ��ָ�����һ����
    // ָ���ų��������ݵĺ�һλ
    int i = start1 + min(len1, K / 2) - 1;
    int j = start2 + min(len2, K / 2) - 1;

    // ��Ϊ���������飬����С��һ������Ԫ���µ�����Ԫ�ض�С�ڴ������Ԫ�أ��Ϳ��Խ�С��Ԫ�����ڵ������Ԫ������λ��֮�µ������ų���֮���ٴν����ų���ֱ������һ��������������k=1��
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
    // �������������Ϊ����ż�������⣬
    // ��������1+2��/2 = 1����1+1��/2 = 1��
    // ż������2+1��/2 = 1����2+2��/2 = 2��
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

    // ÿ��ѭ��ѡ��һ�����ģ�����������չ���ж������ַ��Ƿ���ȡ�
    for (int i = 0; i < strlen(s); i++)
    {
        // ��Ϊ�����������ַ�����ż�����ַ�����������Ҫ��һ���ַ���ʼ��չ���ߴ������游֮�俪ʼ��չ������һ����(n+n-1)������
        int len1 = expandAroundCenter(s, i, i);
        int len2 = expandAroundCenter(s, i, i + 1);
        int len = max(len1, len2);

        if (len > end - start)
        {
            start = i - (len - 1) / 2;
            end = i + len / 2;
        }

        /***********************
         * �����ƶ�˳��
         *[a,b,b,a,b,a]==>[(a),b,b,a,b,a]==>[a(,)b,b,a,b,a]
         *����start��end
         *[{a,b}(,){b,a},b,a]      [a,b,{b},(a),{b},a]
         *start = i - (len - 1) / 2��
         *�����ż�����ַ�����ʱ��i��ʾ���Ŀ�λǰһ���ַ���������ʱ���ʾ�����ַ���
         *end = i + len / 2;
         *��ʾ����λ������len/2���ȵ��ַ�λ��
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
    // �Ƚ�����Ҫ���е�����ֱ�ӷ��ء�
    if (len < numRows || numRows == 1)
        return s;

    char *convertS = (char *)malloc(sizeof(char) * (len + 1));
    memset(convertS, 0, sizeof(char) * (len + 1));

    int idx = 0;
    for (int i = 1; i <= numRows; ++i)
    {
        if (i == 1 || i == numRows)
        {
            // ��0��͵�numRows-1����±�������step ��step=2*numRows-2
            int tmp = (numRows - 1) * 2;
            printf("tmp:%d\n", tmp);
            for (int j = i - 1; j < len; j += tmp)
            {
                convertS[idx++] = s[j];
            }
        }
        else
        {
            // �м����±�������(step-2*����)��(2*����)���档
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
    int flag = 0; // 0����

    // �����ո�
    while (s[index] == ' ')
    {
        index++;
    }
    // �ж�����
    if (s[index] == '+' || s[index] == '-')
    {
        if (s[index] == '-')
            flag = 1;

        index++;
    }

    // ��������
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

/*����һ���������� nums ��һ������ k ��
ÿһ�������У�����Ҫ��������ѡ����Ϊ k ���������������������Ƴ����顣
��������Զ�����ִ�е����������� */
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
 *����������array�в���һ��λ�ã�ʹ������Ԫ������������ߵݼ���Ҫ��
 *�����������Ҳ������ߵĵײ����߶��� */
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

    // �ȴ�С��������
    quickSort(nums, 0, numsSize - 1);

    for (int a = 0; a < numsSize - 2; a++)
    {
        // aλ�õ�ֵ����С��
        if (nums[a] > 0)
            break;

        // ȥ��
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

                // ȥ��
                while (c > b && nums[c] == nums[c - 1])
                    c--;

                // ȥ��
                while (c < b && nums[b] == nums[b + 1])
                    b++;

                printf("d0:%d\td1:%d\td2:%d\n", nums[a], nums[b], nums[c]);
                (*returnColumnSizes)[*returnSize] = 3;
                (*returnSize)++;
                if ((*returnSize) >= base)
                {
                    base *= 2; // base += 2;ʱ���ڴ�ᳬ������
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
        // ���˵���ͬ��ֵ
        if (i > 0 && nums[i] == nums[i - 1])
            continue;

        int left = i + 1;
        int right = numsSize - 1;
        while (left < right)
        {
            int sum = nums[i] + nums[left] + nums[right];
            // �Ѿ��������ֵ��
            if (sum == target)
                return target;

            // �Ƚϻ��С��ֵ
            res = fabs(sum - target) > fabs(res - target) ? res : sum;

            if (sum < target)
            {
                left++; // ����left��Ӧ������sum�Ĵ�С
                while (left < right && nums[left - 1] == nums[left])
                    ++left;
            }
            else
            {
                right--; // ��Сright��Ӧ�ļ�Сsum�Ĵ�С
                while (left < right && nums[right + 1] == nums[right])
                    --right;
            }
        }
    }

    return res;
}

/* �ڻ����㷨�У�������Ҫ�Ե�ǰ�ڵ���в���������һ����ȱ��������ӽڵ㡣�����Ǳ��������е��ӽڵ����Ҫ���ݵ���ǰ�ڵ�ĸ��ڵ㣬�������������Ҫ������ǰ�ڵ�Ĳ�����
����δ����У�pathIndex ��ʾ path ��������һ��Ҫ�����ַ�λ�á���ÿ�ν���ݹ�ʱ��������Ҫ�� path
�����е���һ���ַ�������䣬���� pathIndex ���� 1��Ȼ��ݹ������һ�㣬�ȵݹ鷵��ʱ��Ҫ���л��ݣ��� pathIndex ��ȥ
1�������� path ��������*/
// ������Ӧ��ĸ��
char phoneMap[11][5] = {"\0", "\0", "abc\0", "def\0", "ghi\0", "jkl\0", "mno\0", "pqrs\0", "tuv\0", "wxyz\0"};
void backTrackCombination(char **result, int *resultIndex, char *path, int *pathIndex, char *digits, int startIndex,
                          int len)
{
    /* startIndex == len ���˳��ݹ���������� startIndex ���� len
     * ʱ��˵���Ѿ���ϳ�һ�������ĵ绰���룬������뵽��������У��������ݹ顣��֮ǰ�ĵݹ�����У�ÿ����ȱ����Ķ��ǵ�ǰ
     * digits �����е���һ�����֣�ֱ����ϳ�һ�������ĵ绰����Ϊֹ�� */
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
    /* pathIndex ��ʾ path ��������һ��Ҫ�����ַ�λ�á���ÿ�ν���ݹ�ʱ��������Ҫ�� path
     * �����е���һ���ַ�������䣬���� pathIndex ���� 1��Ȼ��ݹ������һ�㣬�ȵݹ鷵��ʱ��Ҫ���л��ݣ��� pathIndex
     * ��ȥ 1�������� path �������䡣�ڵݹ�������޸��� path �����е�ֵ���ڻ��ݵ���ǰ�ڵ�ʱ��������Ҫ�� path
     * ����ָ����ڽ��뵱ǰ�ڵ�ʱ��״̬�� */
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
                // ��ֹsumֵ����int��Χ
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
                        base *= 2; // base += 2;ʱ���ڴ�ᳬ������
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
/* ����true���������"[]","[]()","{[]}()"
 *false �����"[","{[}]","]","{[}","{{}(})"
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

/*  �ϲ�������������
    �ݹ�ϲ��ڵ㣬��ǰ�ڵ�˭С�����������С�Ľڵ��next����һ����������ݹ�ϲ���ֱ������������һ����nxet�������ˣ��Ǿ�û���ָ������ˣ�ֻ�ܷ���
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

// �ϲ������������
/*���η������������ϲ�
 *����1������2������3������4������5��
 *[����1������2��][����3������4������5��]
 *[����1������2��][[����3��][����4������5��]]
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

/* �������������еĽڵ�
�ݹ����ֹ������������û�нڵ㣬����������ֻ��һ���ڵ㣬��ʱ�޷����н�����
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

/* ��ת����
    �ݹ鷨��1step:
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

// �������������
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

// �ַ������鷴ת
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
 * ��ʮ����������Ϊ�������ַ�������  7==��"111"
 *
 * return :���ص�������calloc���ٿռ䣬��Ҫ����free();
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
 * k��һ�鷭ת����,���õݹ�ķ���
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

    // ֻ������k���ڵ�����Ž��з�ת
    if (count == k)
    {
        // �ȱ���������k������һ�������飬�����һ��k������ת�����ط�ת��ı�ͷ��֮��Ӻ���ǰ���η�תk������
        cur = reverseKGroup(cur, k);
        while (count)
        {
            struct ListNode *tmp = head->next;
            head->next = cur; // ���Ӻ�һ�������鷴ת���ͷ�ڵ㣬����NULL�����߲�����k���ڵ��������ͷ�ڵ�
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
 * ɾ�����������е��ظ���
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

/*�Ƴ�Ŀ��Ԫ�� */
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

// ����
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

// ����ߵ�
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
 * 31����һ������
 * example:
 * 4,5,2,6,3,1              ==>start
 * step 1:  4,5,(2),6,3,1   ==>��С��2,�Ӻ���ǰ�����������ĵ�һ��Ԫ��
 * step 2:  4,5,(2),5,(3),1 ==>�ϴ���3������Ԫ���бȽ�С��2���Ԫ������С��Ԫ��
 * step 3:  4,5,(3),5,(2),1 ==>swep�ϴ����ͽ�С��
 * step 4:  4,5,(3),(5,2,1) ==>�ߵ�������Ľϴ�������Ԫ�ص�����
 * step 5:  4,5,(3),(1,2,5) ==>�ߵ�������Ľϴ�������Ԫ�ص�����
 * 4,5,3,1,2,5              ==>end
 *  */
void nextPermutation(int *nums, int numsSize)
{
    int i = numsSize - 2;
    // Ѱ��һ����ߵĽ�С������С����������ǽ�������
    while (i >= 0 && nums[i] >= nums[i + 1])
    {
        i--;
    }

    printf("i:%d\n", i);

    if (i >= 0)
    {
        int j = numsSize - 1;
        // Ѱ���ұߵĽϴ������ϴ���������С���Ƚ�С������Ԫ���бȽ�С���������Ԫ�ض�ҪС
        while (j >= 0 && nums[i] >= nums[j])
        {
            j--;
        }
        // �����ϴ����ͽ�С����λ��
        swap(nums + i, nums + j);
    }

    // ��������Ľϴ��������������ߵ�����Ϊ����Ϊ֮�������㽵��ֱ�ӵߵ�˳��Ϳ��Եõ�һ����С������
    permutationReverse(nums, i + 1, numsSize - 1);
}
/*******
 * ��������λ��
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
        // ������9������1����ѭ��
        if (digits[i] != 9)
        {
            digits[i]++;
            break;
        }
        // ����9��1��0
        digits[i] = 0;
    }

    // ������λΪ0����ʾ��Ϊȫ9��ɵ�������Ҫ����ռ�
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

// ���������
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
 * ������ת��������
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
 * �����������в���Ԫ�صĵ�һ�������һ��λ��
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
 * �����ж��Ƿ��л�
 * ����ָ���ظ��������ָ�����ָ��������˵�����ڻ�
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
 * �ж������Ƿ���ڻ������ڵĻ������뻷�ĵ�һ���ڵ�
 * fast = 2* slowǰ���£�fast��slow��Ȧ��һȦ����������
 * 1����ָ���ߵ�����ָ�������
 * 2����ָ���߹���·����ָ���߹�һ��
 * 3����ָ���߹���ʣ��·�̣�Ҳ���Ǻ���ָ���߹���ȫ��·����ȡ�
 * 4����ȥ��ָ��׷����ָ��İ�Ȧ��ʣ��·�̼�Ϊ�����뻷���롣
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
 * ����������
 * ��̬�滮������˳�����ӵ�˳���������ܼӵ������ֵ��������֮ǰ��ֵ���Ƚϣ����������ֵ��
 * ���ǰ����ۼ�������û�б���󣬾Ͷ���ǰ����ۼ����ݣ����������ֹ������ֵ��
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
 * �ַ������
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

    // ���鷴ת
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
 * ��ת�ַ���
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
 * ��¥�ݣ���̬�滮����
 * f(x) = f(x-1)+f(x-2)
 * ÿ��ֻ����1��̨�׻�������̨�ף�������n��̨�׵ķ���������������n-1��̨�׵ķ�������������n-2��̨�׵ķ�����֮�͡�
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
 * 쳲���������
 * F(0) = 0;F(1) = 1
 * F(N) = F(N-1) + F(N-2) N > 1
 * ���׿�������ѧ�Ƶ����ƹ�ʽ�������������㡣
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
 * �ƶ��㣬�������е�0�ŵ�������治Ӱ����������ַ�˳��
 * ���ƿ���˼�룬��0Ϊ�м�㣬�Ѳ�����0�ķŵ��м����ߣ��ѵ���0�ķŵ��м���ұ�
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
 * �Ƴ�Ԫ�أ�ɾ��������ָ��Ԫ��
 * ����ָ����������ָ�룺Ѱ�Ҳ�����Ŀ��Ԫ�ص�ָ��
 *              ��ָ�룺ָ���ָ����Ѱ����Ԫ�ظ��ǵ�λ��
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
 * ������ͼ�������ٲ�������
 * ��������һ��Ԫ�ؼ��룬֮����ͣ��������ͱȳ�ʼ����͵�һ��Ҫ�󣬼����Բ������������ѡ��һ��Ԫ�ؼ��������ͱȽϲ�����ֱ�������С��ԭʼ����͵�һ�룬��������Ҫ�������Ρ�
 * ������ÿ�ζ�����һ��Ԫ����������������ġ�
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
 * ������С��������
 * ����һ������ n ���������������һ�������� target ��
 * �ҳ���������������� �� target �ĳ�����С�� ���������� [numsl, numsl+1, ..., numsr-1, numsr]
 * ���������䳤�ȡ���������ڷ��������������飬���� 0 ��
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

    // ��������
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
 * ���ֲ���
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
 * �ϲ�������������
 * ���������� �ǵݼ�˳�� ���е��������� nums1 �� nums2�������������� m �� n ���ֱ��ʾ nums1 �� nums2 �е�Ԫ����Ŀ��
 * ���� �ϲ� nums2 �� nums1 �У�ʹ�ϲ��������ͬ���� �ǵݼ�˳�� ���С�
 * ע�⣺���գ��ϲ������鲻Ӧ�ɺ������أ����Ǵ洢������ nums1 �С�Ϊ��Ӧ�����������nums1 �ĳ�ʼ����Ϊ m + n������ǰ m
 * ��Ԫ�ر�ʾӦ�ϲ���Ԫ�أ��� n ��Ԫ��Ϊ 0 ��Ӧ���ԡ�nums2 �ĳ���Ϊ n ��
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
 * ��Ծ��Ϸ
 * ����һ���Ǹ��������� nums �������λ������� ��һ���±� ��
 * �����е�ÿ��Ԫ�ش������ڸ�λ�ÿ�����Ծ����󳤶ȡ�
 * �ж����Ƿ��ܹ��������һ���±ꡣ
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
        // ��ʾ��������������i��λ��
        if (i > cover)
            return false;

        // ��¼��ÿ������������Զ����
        cover = fmax(cover, i + nums[i]);

        if (cover >= (numsSize - 1))
            return true;
    }
    return true;
}

/**************
 * ������Ʊ�����ʱ��
 * ����һ������ prices �����ĵ� i ��Ԫ�� prices[i] ��ʾһ֧������Ʊ�� i ��ļ۸�
 * ��ֻ��ѡ�� ĳһ�� ������ֻ��Ʊ����ѡ���� δ����ĳһ����ͬ������ �����ù�Ʊ�����һ���㷨�����������ܻ�ȡ���������
 * ��������Դ���ʽ����л�ȡ�������������㲻�ܻ�ȡ�κ����󣬷��� 0 ��
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
            // �����������
            if (maxProfit < (prices[fast] - prices[low]))
                maxProfit = prices[fast] - prices[low];

            fast++;
        }
        else
        {
            // ��lowԪ�ش���fastԪ��ʱ�򣬺�����ֵ�Ԫ�ؼ�ȥfastԪ�ص�����϶��ȼ�ȥlowԪ�ص�����������������ֱ�Ӵ�fastԪ�ؿ�ʼ
            low = fast;
            fast += 1;
        }
    }

    return maxProfit;
}

/***********
 * x��ƽ����
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
 * ����ˮ����
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
 * ���������������
 *  ���1���������������͵�������Ϊ nums[i:j]������ nums[i] ��nums[j?1] ��j?i ��Ԫ�أ����� 0��i<j��n
 *  ���2���������������͵�������Ϊ nums[0:i] ��nums[j:n]������ 0<i<j<n
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
        // �Ҷ˵����귶Χ��[0,i]�����ǰ׺�Ϳ�����leftMax[i]
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
//���ο����ƹ㿴��һ��˫�����ȵ���������ͣ�
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
    // ������������������� = ��ͨ����� - ��ͨ�������С�������
    int preMax = nums[0];
    int maxRes = nums[0]; // ��ͨ���������������
    int preMin = nums[0];
    int minRes = nums[0]; // ��ͨ�������С�������
    int sum = nums[0];
    for (int i = 1; i < numsSize; i++)
    {
        // ������ͨ���������������
        preMax = fmax(preMax + nums[i], nums[i]);
        maxRes = fmax(maxRes, preMax);
        // ������ͨ�������С�������
        preMin = fmin(preMin + nums[i], nums[i]);
        minRes = fmin(minRes, preMin);
        // ������ͨ����ĺ�
        sum += nums[i];
    }

    // maxRes<0,��ʾ�����в��������ڵ���0��Ԫ�أ�minRes�����������е�����Ԫ�أ�ʵ��ȡ����������Ϊ�գ��������������;���maxRes��������һ������Ԫ�ء�
    if (maxRes < 0)
        return maxRes;
    else
        return fmax(maxRes, sum - minRes);
}
/**********
 * ����������͵ľ���ֵ�����ֵ
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