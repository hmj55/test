#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Ѱ��������Сֵ���±�
int argmin(int *index, int index_len)
{
    int min_index = 0;
    int min = index[0];
    for (int i = 1; i < index_len; i++)
    {
        if (index[i] < min)
        {
            min = index[i];
            min_index = i;
        }
    }
    return min_index;
}

/**
 * @brief Ѱ�Ҳ��庯��
 *
 * @param data ������ݵ�����
 * @param index ��ŷ�ֵ���±������
 * @param len_index ��ֵ��������index���鳤��
 */
void AMPD(float *data, int *index, int *len_index)
{
#define DATA_LEN 50
    int size = DATA_LEN;
    int p_data[DATA_LEN] = {0};
    int arr_rowsum[DATA_LEN] = {0};
    int min_index, max_window_length;
    *len_index = 0;

    for (int i = 0; i < size; i++)
    {
        p_data[i] = 0;
    }
    for (int k = 1; k < size / 2 + 1; k++)
    {
        int row_sum = 0;
        for (int i = k; i < size - k; i++)
        {
            if ((data[i] > data[i - k]) && (data[i] > data[i + k]))
                row_sum -= 1;
        }
        *(arr_rowsum + k - 1) = row_sum;
    }

    min_index = argmin(arr_rowsum, size / 2); // �˴�Ϊ���Ĵ���
    max_window_length = min_index;

    for (int k = 1; k < max_window_length + 1; k++)
    {
        for (int i = 1; i < size - k; i++)
        {
            if ((data[i] > data[i - k]) && (data[i] > data[i + k]))
                p_data[i] += 1;
        }
    }

    for (int i_find = 0; i_find < size; i_find++)
    {
        if (p_data[i_find] == max_window_length)
        {
            index[*len_index] = i_find;
            (*len_index) += 1;
        }
    }
}

/**
 * @brief �Ƴ�float�����������ظ�Ԫ����һ�����ɺ���Ԫ��ǰ�Ʋ���
 *
 * @param value ��Ҫ���������
 * @param len  ���鳤��
 * @param res   ���մ���������
 */
void removeDuplicatesfloat(float *value, int len, float *res)
{
    int retSize = 0;
    for (int i = 0; i < len; i++)
    {
        if (retSize > 0 && res[retSize - 1] == value[i])
        {
            retSize = retSize;
        }
        else
        {
            res[retSize++] = value[i];
        }
    }
    res[retSize] = 0.0f;
}
int main(int argc, const char *argv[])
{
#define VALUE_LEN (50)
    float value2[VALUE_LEN] = {
        5, 15, 25, 35, 45, 55, 65, 75, 85, 95, 85, 75, 65, 55, 45, 35, 25, 15, 5, 0, 5, 15, 25, 35, 45, 55, 65, 75, 85, 95, 85, 75, 65, 55, 45, 35, 25, 15, 5, 0, 5, 15, 25, 35, 45, 55, 65, 75, 85, 95};
    int index[VALUE_LEN] = {0};
    int lenIndexNum = 0;

    float removedValues[VALUE_LEN] = {0};
    removeDuplicatesfloat(value2, VALUE_LEN, removedValues); // ȥ�������ظ�Ԫ���е�һ��

    AMPD(removedValues, index, &lenIndexNum); // ���ҳ�����
    printf("lenIndexNum = %d\n", lenIndexNum);

    for (int i = 0; i < lenIndexNum; i++)
    {
        printf("maxValue index[%d]:%f\t", index[i], removedValues[index[i]]);
    }
    printf("\n");

    // ��ת����
    for (int i = 0; i < VALUE_LEN; i++)
    {
        removedValues[i] *= -1;
    }

    AMPD(removedValues, index, &lenIndexNum); // ���ҳ���ת������Ĳ��壬Ҳ����ԭ����Ĳ���
    printf("lenIndexNum = %d\n", lenIndexNum);

    for (int i = 0; i < lenIndexNum; i++)
    {
        printf("minValue index[%d]:%f\t", index[i], -removedValues[index[i]]);
    }
    return 0;
}