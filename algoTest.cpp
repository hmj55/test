#include <algorithm>
#include <ctime>
#include <iostream>
#include <numeric>
#include <stdio.h>
#include <string.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>
using namespace std;

struct ListNode
{
    int val;
    ListNode *next;
    ListNode() : val(0), next(nullptr){};
    ListNode(int x) : val(x), next(nullptr){};
    ListNode(int x, ListNode *next) : val(x), next(next){};
};

class Solution
{
    vector<int> tmp;
    void mergeSort(vector<int> &nums, int l, int r)
    {
        if (l >= r)
            return;
        int mid = (l + r) / 2;
        mergeSort(nums, l, mid);
        mergeSort(nums, mid + 1, r);

        int i = l;
        int j = mid + 1;
        int cnt = 0;
        while (i <= mid && j <= r)
        {
            if (nums[i] <= nums[j])
                tmp[cnt++] = nums[i++];
            else
                tmp[cnt++] = nums[j++];
        }

        while (i <= mid)
        {
            tmp[cnt++] = nums[i++];
        }

        while (j <= r)
        {
            tmp[cnt++] = nums[j++];
        }

        for (int i = 0; i < r - l + 1; ++i)
        {
            nums[i + l] = tmp[i];
        }
    }

    int partition(vector<int> &nums, int l, int r)
    {
        int pivot = nums[r];
        int i = l - 1;

        for (int j = l; j <= r - 1; ++j)
        {
            if (nums[j] <= pivot)
            {
                i = i + 1;
                swap(nums[i], nums[j]);
            }
        }
    }

    int randomized_partition(vector<int> &nums, int l, int r)
    {
        int i = rand() % (r - l + 1) + l;
        swap(nums[r], nums[i]);
        return partition(nums, l, r);
    }

    void randomized_quicksort(vector<int> &nums, int l, int r)
    {
        if (l < r)
        {
            int pos = randomized_partition(nums, l, r);
            randomized_quicksort(nums, l, pos - 1);
            randomized_quicksort(nums, pos + 1, r);
        }
    }

  public:
    // ������м�ڵ�
    ListNode *middleNode(ListNode *head)
    {
        ListNode *slow = head;
        ListNode *fast = head;
        while (fast != nullptr && fast->next != nullptr)
        {
            slow = slow->next;
            fast = fast->next->next;
        }
        return slow;
    }

    // ��ת����
    ListNode *reverseList(ListNode *head)
    { // ����
        ListNode *prev = nullptr;
        ListNode *curr = head;
        while (curr)
        {
            ListNode *next = curr->next;
            curr->next = prev;
            prev = curr;
            curr = next;
        }
        return prev;
    }

    ListNode *reverseListEx(ListNode *head)
    { // �ݹ�
        if (!head || !head->next)
            return head;

        ListNode *newHead = reverseListEx(head->next);
        head->next->next = head;
        head->next = nullptr;
        return newHead;
    }

    // �ϲ�����,����ϲ�
    void mergeList(ListNode *l1, ListNode *l2)
    {
        ListNode *l1_tmp;
        ListNode *l2_tmp;

        while ((l1 != nullptr & l2 != nullptr))
        {
            l1_tmp = l1->next;
            l2_tmp = l2->next;

            l1->next = l2;
            l1 = l1_tmp;

            l2->next = l1;
            l2 = l2_tmp;
        }
    }

    // �������������Ա�洢�������±�����������
    void reorderList(ListNode *head)
    {
        if (head == NULL)
            return;

        vector<ListNode *> vec;
        ListNode *node = head;

        while (node != NULL)
        {
            vec.emplace_back(node);
            node = node->next;
        }

        int i = 0;
        int j = vec.size() - 1;
        while (i < j)
        {
            vec[i]->next = vec[j];
            i++;
            if (i == j)
                break;
            vec[j]->next = vec[i];
            j--;
        }
        vec[i]->next = nullptr;
    }

    // �������������е���Ϊ�����֣���벿�ַ�ת������ϲ�������
    void reorderListEX(ListNode *head)
    {
        if (head == nullptr)
            return;

        ListNode *mid = middleNode(head);
        ListNode *l1 = head;
        ListNode *l2 = mid->next;
        mid->next = nullptr;
        l2 = reverseList(l2);
        mergeList(l1, l2);
    }

    /************************
     *  ����ѡ���㷨��
     * �ڷֽ�Ĺ��̵��У����ǻ����������л��֣�������ֵõ��� q ���þ���������Ҫ���±꣬��ֱ�ӷ��� a[q]��
     * ������� q ��Ŀ���±�С���͵ݹ��������䣬����ݹ��������䡣
     * �����Ϳ��԰�ԭ���ݹ�����������ֻ�ݹ�һ�����䣬�����ʱ��Ч�ʡ�
     ***********************/
    int quickselect(vector<int> &nums, int l, int r, int k)
    {
        if (l == r)
            return nums[k];

        int partition = nums[l];
        int i = l - 1;
        int j = r + 1;
        while (i < j)
        {
            do
            {
                i++;
            } while (nums[i] < partition);
            do
            {
                j--;
            } while (nums[j] > partition);

            if (i < j)
                swap(nums[i], nums[j]);
        }

        if (k <= j)
            return quickselect(nums, l, j, k);
        else
            return quickselect(nums, j + 1, r, k);
    }

    // �����еĵ�K�����Ԫ��
    int findKthLargest(vector<int> &nums, int k)
    {
        int n = nums.size();
        return quickselect(nums, 0, n - 1, n - k);
    }

    // �ƶ�Ƭ�εõ��ַ���
    /****
     * �������λ�÷�����Ҫ��start�ܱ��target����Ҫ������������������
     * ȥ���»���֮��������Ӧ������ȵģ�
     * start�е�L��target�е�L��һһ��Ӧ�ģ���L���������ƶ�������target�е�i��L���Բ�����start�е�i��L���ұߣ�
     * start�е�R��target�е�R��һһ��Ӧ�ģ���R���������ƶ�������target�е�i��R���Բ�����start�е�i��R�����
     * L������R����ߣ��������ߵ�L��R������Ҫ���
     */
    bool canChange(string start, string target)
    {
        int n = start.length();
        int i = 0;
        int j = 0;
        while (i < n && j < n)
        {
            while (i < n && start[i] == '_')
                i++;

            while (j < n && target[j] == '_')
                j++;

            if (i < n && j < n)
            {
                if (start[i] != target[j])
                    return false;

                char c = start[i];
                if ((c == 'L' && i < j) || (c == 'R' && i > j))
                    return false;

                i++;
                j++;
            }
        }

        // �ж��Ƿ���δ����ġ�L�����ߡ�R'�ַ����еĻ�����false
        while (i < n)
        {
            if (start[i] != '_')
                return false;
            i++;
        }

        while (j < n)
        {
            if (target[j] != '_')
                return false;

            j++;
        }

        return true;
    }

    // ��ҽ���,��̬�滮
    int rob(vector<int> &nums)
    {
        if (nums.size() == 0)
            return 0;

        int N = nums.size();
        vector<int> dp(N + 1, 0);
        dp[0] = 0;
        dp[1] = nums[0];
        for (int k = 2; k <= N; k++)
        {
            dp[k] = max(dp[k - 1], nums[k - 1] + dp[k - 2]); // ���Ż�
        }

        return dp[N];
    }
    // ��ҽ���,��̬�滮,�ռ��Ż���ʵ����ÿ��ֻ�õ�f(n-1)��f(n-2)�Ľ��֮ǰ�Ľ���Ѿ��ò����ˣ��Ͳ��ô洢�ˡ�
    int robEX(vector<int> &nums)
    {
        int prev = 0;
        int curr = 0;

        for (int i : nums)
        {
            int temp = max(curr, prev + i);
            prev = curr;
            curr = temp;
        }

        return curr;
    }

    // ��������˵�������,������������0�ĳ���
    int maxDistToColsest(vector<int> &seats)
    {
        int res = 0;
        int l = 0;
        // ��0��ʼ��1��β
        while (l < seats.size() && seats[l] == 0)
        {
            ++l;
        }
        res = max(res, l);

        while (l < seats.size())
        {
            int r = l + 1;
            while (r < seats.size() && seats[r] == 0)
            {
                ++r;
            }

            /*
            if (r == seats.size())
                res = max(res, r - l - 1); // ��0��β��ǰ��1
            else
                res = max(res, (r - l) / 2); // ����1֮������м��0��λ��
            */

            res = (r == seats.size()) ? max(res, r - l - 1) : max(res, (r - l) / 2);
            l = r;
        }
        return res;
    }

    // ��������˵�������,��λ�������������λ�õ����Ŀ�λ����
    int maxDistToClosestEX(vector<int> &seats)
    {
        int first = -1;
        int last = -1;
        int d = 0;
        int n = seats.size();
        for (int i = 0; i < n; ++i)
        {
            if (seats[i] == 1)
            {
                if (last != -1)
                {
                    d = max(d, i - last); // ��¼����1�м��0����󳤶�
                }
                if (first == -1)
                    first = i; // ��¼0��ʼ1����Ϊ��೤��
                last = i;
            }
        }
        // max(first, n - last - 1)//���Ҳ��1��ʼ0��β�ĳ���
        return max({d / 2, max(first, n - last - 1)});
    }

    // �ҳ�תȦ��Ϸ���
    vector<int> circularGameLosers(int n, int k)
    {
        vector<bool> visit(n, false);
        for (int i = k, j = 0; !visit[j]; i += k)
        {
            visit[j] = true;
            j = (j + i) % n;
        }

        vector<int> ans;
        for (int i = 0; i < n; i++)
        {
            if (!visit[i])
                ans.emplace_back(i + 1);
        }

        return ans;
    }

    // �����������,��̬�滮,ϸ��Ϊ��¼��ÿ��λ�õ������д�С������������һ��������ֵ
    int lengthOfLIS(vector<int> &nums)
    {
        int n = nums.size();
        if (n == 0)
            return 0;

        vector<int> dp(n, 0);
        for (int i = 0; i < n; ++i)
        {
            dp[i] = 1;
            for (int j = 0; j < i; ++j)
            {
                if (nums[j] < nums[i])
                    dp[i] = max(dp[i], dp[j] + 1);
            }
        }
        return *max_element(dp.begin(), dp.end());
    }

    /****
     * �����������:̰��+���ֲ���
     *  �赱ǰ�����������������еĳ���Ϊ len����ʼʱΪ 1������ǰ�����������nums���ڱ�����nums[i] ʱ��
     * ���nums[i]>d[len] ����ֱ�Ӽ��뵽 d ����ĩβ�������� len=len+1��
     * ������ d �����ж��ֲ��ң��ҵ���һ���� nums[i] С���� d[k] �������� d[k+1]=nums[i]��
     */
    int lenghtOfLISEX(vector<int> &nums)
    {
        int len = 1;
        int n = nums.size();
        if (n == 0)
            return 0;

        vector<int> d(n + 1, 0);
        d[len] = nums[0];
        for (int i = 1; i < n; ++i)
        {
            if (nums[i] > d[len])
            {
                d[++len] = nums[i];
            }
            else
            {
                int l = 1;
                int r = len;
                int pos = 0;
                while (l <= r)
                {
                    int mid = (l + r) >> 1;
                    if (d[mid] < nums[i])
                    {
                        pos = mid;
                        l = mid + 1;
                    }
                    else
                    {
                        r = mid - 1;
                    }
                }
                d[pos + 1] = nums[i];
            }
        }
        return len;
    }

    // ����Ԫ�أ�ָ�������г��ִ�������[n/2]��Ԫ��,��ϣ��
    int majorityElement(vector<int> &nums)
    {
        unordered_map<int, int> counts;
        int majority = 0;
        int cnt = 0;
        for (int num : nums)
        {
            ++counts[num];
            if (counts[num] > cnt)
            {
                majority = num;
                cnt = counts[num];
            }
        }
        return majority;
    }

    // ����Ԫ������������Ԫ��һ��������������м�һ��Ԫ�أ���Ϊ����Ԫ�ش�������[n/2]
    int majorttyElementEX(vector<int> &nums)
    {
        sort(nums.begin(), nums.end());
        return nums[nums.size() / 2];
    }

    /**********
     *  ����Ԫ�أ�Boyer-MooreͶƱ�㷨��
     * ��ÿһ��ͶƱ�����У���������ɾ��������ͬ��Ԫ�أ�ֱ��ͶƱ�����޷���������ʱ����Ϊ�ջ���������ʣ�µ�Ԫ�ض���ȡ�
     * ��Ϊ����Ԫ����������ռ�г���һ�룬�������ÿ��ȥ��������ͬ��Ԫ�أ�ֱ��û�в�ͬ��Ԫ�ء�
     * ��ʱ����Ϊ�ջ���ȫΪ���Ԫ�أ�Ϊ�ձ�ʾ�����ڶ���Ԫ�أ�ʣ�µ�Ԫ�ض���ȱ�ʾ���Ԫ�ؾ��Ƕ���Ԫ��
     */
    int majorityElementEx1(vector<int> &nums)
    {
        int candidate = -1;
        int count = 0;
        for (int num : nums)
        {
            if (num == candidate)
                ++count;
            else if (--count < 0)
            {
                candidate = num;
                count = 1;
            }
        }

        return candidate;
    }

    //   ���λ�ÿ�������
    vector<int> sortArray(vector<int> &nums)
    {
        srand((unsigned)time(NULL));
        randomized_quicksort(nums, 0, (int)nums.size() - 1);
        return nums;
    }

    //   �鲢����
    vector<int> sortArrayEX(vector<int> &nums)
    {
        tmp.resize((int)nums.size(), 0);
        mergeSort(nums, 0, (int)nums.size() - 1);
        return nums;
    }

    //   ���������
    int longestConsecutive(vector<int> &nums)
    {
        // ��������set�������numsԪ��
        unordered_set<int> num_set;
        for (const int &num : nums)
        {
            num_set.insert(num);
        }

        int longestStreak = 0;

        for (const int &num : num_set)
        {
            // ����ǰһ������������ʾ�Ѿ���¼������������û��Ҫ�ٽ���һ�μ���
            if (!num_set.count(num - 1))
            {
                int currentNum = num;
                int currentStreak = 1;

                while (num_set.count(currentNum + 1))
                {
                    currentNum += 1;
                    currentStreak += 1;
                }
                longestStreak = max(longestStreak, currentStreak);
            }
        }
        return longestStreak;
    }

    // �ϲ�����
    vector<vector<int>> merge(vector<vector<int>> &intervals)
    {
        if (intervals.size() == 0)
            return {};

        sort(intervals.begin(), intervals.end());
        vector<vector<int>> merged; // ����һ����ά����

        for (int i = 0; i < intervals.size(); i++)
        {
            int left = intervals[i][0];
            int right = intervals[i][1];

            if (!merged.size() || merged.back()[1] < left)
            {
                merged.push_back({left, right});
                // �洢���䣬���ߺ���������ǰ�治����
            }
            else
            {
                merged.back()[1] = max(merged.back()[1], right);
                // �������һ������Ľ���λ�ã������������һ������Ľ���λ���ں�һ���ж��������ʼλ��֮��
            }
        }
        return merged;
    }

    //  �½�·����С�ͣ�������������н����������С���
    int minFallingPathSum(vector<vector<int>> &matrix)
    {
        int n = matrix.size();
        vector<vector<int>> dp(n, vector<int>(n));
        copy(matrix[0].begin(), matrix[0].end(), dp[0].begin());

        for (int i = 1; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                int mn = dp[i - 1][j];
                if (j > 0)
                    mn = min(mn, dp[i - 1][j - 1]); // ��ֹ��ȡ����Խ�����
                if (j < n - 1)
                    mn = min(mn, dp[i - 1][j + 1]); // ��ֹ�Ҳ�����Խ��

                dp[i][j] = mn + matrix[i][j]; // �Ϸ�Ԫ��+�·���������תԪ������С��һ��
            }
        }

        return *min_element(dp[n - 1].begin(), dp[n - 1].end()); // ���м���������С��һ��
    }

    // ��Ծ��ϷII
    int jump(vector<int> &nums)
    {
#if 0
        int ans = 0;
        int start = 0;
        int end = 1;

        while (end < nums.size())
        {
            int maxPos = 0;
            for (int i = start; i < end; i++)
            {
                // ����������Զ�ľ���
                maxPos = max(maxPos, i + nums[i]);
            }
            start = end;      // ��һ�������㷶Χ��ʼ�ĸ���
            end = maxPos + 1; // ��һ�������㷶Χ�����ĸ���
            ans++;            // ��Ծ����
        }
        return ans;
#endif
        int ans = 0;
        int end = 0;
        int maxPos = 0;
        for (int i = 0; i < nums.size(); i++)
        {
            maxPos = max(nums[i] + i, maxPos);
            if (i == end)
            {
                end = maxPos;
                ans++;
            }
        }
        return ans;
    }

    string findReplaceStringEX(string s, vector<int> &indices, vector<string> &sources, vector<string> &targets)
    {
        int n = s.size();
        int m = indices.size();

        vector<int> ops(m);

        iota(ops.begin(), ops.end(), 0);
        sort(ops.begin(), ops.end(), [&](int i, int j) { return indices[i] < indices[j]; });

        string ans;
        int pt = 0;
        for (int i = 0; i < n;)
        {

            while (pt < m && indices[ops[pt] < i])
            {
                ++pt;
            }

            bool succeed = false;

            while ((pt < m && indices[ops[pt]] == i))
            {
                if (s.substr(i, sources[ops[pt]].size()) == sources[ops[pt]])
                {
                    succeed = true;
                    break;
                }
                ++pt;
            }

            if (succeed)
            {
                ans += targets[ops[pt]];
                i += sources[ops[pt]].size();
            }
            else
            {
                ans += s[i];
                ++i;
            }
        }

        return ans;
    }
    /*******
     * 1,�� s ����Ϊ n������һ����Ϊ n �� replace �б�
     * 2,����ÿ���滻���������ڵ� i ���滻�����������indices[i] ��ʼ���ַ�����ǰ׺ sources[i]��������滻�� target[i]��
     *   ����s="abcd"��s[1:]="bcd" ��ǰ׺ "bc"��
     *   ��ʱ��¼replace[indices[i]]=(target[i],len(sources[i]))����ʾ�滻����ַ������Լ����滻�ĳ��ȡ�
     *
     * 3,��ʼ�� i=0����� replace[i] �ǿյģ���ô�����滻���� s[i]����𰸣�Ȼ�� i ��һ��
     *  ��� replace[i] ��Ϊ�գ���ô��replace[i][0] ����𰸣�Ȼ�� i ����replace[i][1]��ѭ��ֱ�� i=n Ϊֹ��
     */
    string findReplaceString(string s, vector<int> &indices, vector<string> &sources, vector<string> &targets)
    {
        int n = s.length();
        vector<pair<string, int>> replace(n, {"", 1});

        for (int i = 0; i < indices.size(); i++)
        {
            if (s.compare(indices[i], sources[i].length(), sources[i]) == 0)
                replace[indices[i]] = {targets[i], sources[i].length()};
        }

        string ans;

        for (int i = 0; i < n; i += replace[i].second)
        {
            if (replace[i].first.empty())
                ans += s[i];
            else
                ans += replace[i].first;
        }
        return ans;
    }

    //   ��ת���飬ʹ�ö����������洢����
    void rotate(vector<int> &nums, int k)
    {
        int n = nums.size();
        vector<int> newArr(n);

        for (int i = 0; i < n; ++i)
        {
            newArr[(i + k) % n] = nums[i];
        }

        nums.assign(newArr.begin(), newArr.end());
    }

    void rotateEx(vector<int> &nums, int k)
    {
        int n = nums.size();
        k = k % n;
        int count = __gcd(k, n);
        for (int start = 0; start < count; ++start)
        {
            int current = start;
            int prev = nums[start];
            do
            {
                int next = (current + k) % n;
                swap(nums[next], prev);
                current = next;
            } while (start != current);
        }
    }

    // ��ת����
    void reverse(vector<int> &nums, int start, int end)
    {
        while (start < end)
        {
            swap(nums[start], nums[end]);
            start += 1;
            end -= 1;
        }
    }
    /****
     * ���ǿ����Ƚ�����Ԫ�ط�ת������β���� k?mod?n��Ԫ�ؾͱ���������ͷ����Ȼ�������ٷ�ת [0,k?mod?n?1]�����Ԫ�غ�
     * [k?mod?n,n?1]�����Ԫ�ؼ��ܵõ����Ĵ𰸡�
     * */
    void rotateEx1(vector<int> &nums, int k)
    {
        k %= nums.size();
        reverse(nums, 0, nums.size() - 1);
        reverse(nums, 0, k - 1);
        reverse(nums, k, nums.size() - 1);
    }

    /********
     * ��ת��Ƭ��Ϸ
     * ���һ�ſ�Ƭ������������ͬ�����֣���ô���ſ�Ƭ������ô��ת�����涼��������֣�������ּ������������ѡ������ x��
     * �������˼·���������ȱ������п�Ƭ�������Ƭ�ϵ�����������ͬ��������ϣ���� same
     * �У����˼�������������֣������Ա�ѡ�� x������ֻ��Ҫ�ٴα����������֣��ҵ���Сֵ���ɡ�
     * ������Ƿ����ҵ�����Сֵ�����û���򷵻� 0��
     */
    int flipgame(vector<int> &fronts, vector<int> &backs)
    {
        int res = INT_MAX;
        int n = fronts.size();

        unordered_set<int> same;

        for (int i = 0; i < n; ++i)
        {
            if (fronts[i] == backs[i])
                same.insert(fronts[i]);
        }

        for (int &x : fronts)
        {
            if (x < res && same.count(x) == 0)
                res = x;
        }

        for (int &x : backs)
        {
            if (x < res && same.count(x) == 0)
                res = x;
        }

        return res % INT_MAX;
    }

    /**
     * @brief ��Ч������
     *  �����ж�һ�� 9 x 9 �������Ƿ���Ч��ֻ��Ҫ �������¹��� ����֤�Ѿ�����������Ƿ���Ч���ɡ�
     * ���� 1-9 ��ÿһ��ֻ�ܳ���һ�Ρ�
     * ���� 1-9 ��ÿһ��ֻ�ܳ���һ�Ρ�
     * ���� 1-9 ��ÿһ���Դ�ʵ�߷ָ��� 3x3 ����ֻ�ܳ���һ�Ρ�
     *
     * @param board  ��Ҫ�жϵ�Ŀ������
     * @return true ����������Ѿ������������Ч
     * @return false ����������Ѿ������������Ч
     */
    bool isValidSudoku(vector<vector<char>> &board)
    {
        int rows[9][9];
        int cloumns[9][9];
        int subboxes[3][3][9];

        memset(rows, 0, sizeof(rows));
        memset(cloumns, 0, sizeof(cloumns));
        memset(subboxes, 0, sizeof(subboxes));

        for (int i = 0; i < 9; i++)
        {
            for (int j = 0; j < 9; j++)
            {
                char c = board[i][j];
                if (c != '.')
                {
                    int index = c - '0' - 1;
                    rows[i][index]++;
                    cloumns[j][index]++;
                    subboxes[i / 3][j / 3][index]++;
                    if (rows[i][index] > 1 || cloumns[j][index] > 1 || subboxes[i / 3][j / 3][index] > 1)
                        return false;
                }
            }
        }
        return true;
    }

  private:
    bool line[9][9];     // ��
    bool column[9][9];   // ��
    bool block[3][3][9]; // �Ź���
    bool valid;
    vector<pair<int, int>> spaces; // ���ڴ������Ԫ�ص�����

  public:
    /**
     * @brief ��������Ψһ��Ļ��ݺ���
     * ͨ���ж���������ĵ�ǰ�У��к;Ź����Ƿ���֣����ݹ���ݳ�Ψһ��
     *
     * @param board Ŀ������
     * @param pos ���������ֵ�λ��
     */
    void dfsSolveSudoku(vector<vector<char>> &board, int pos)
    {
        if (pos == spaces.size())
        {
            valid = true;
            return;
        }

        auto [i, j] = spaces[pos];
        for (int digit = 0; digit < 9 && !valid; ++digit)
        {
            if (!line[i][digit] && !column[j][digit] && !block[i / 3][j / 3][digit]) // �ж�����������Ƿ���Ч
            {
                // ������һ������������ǰλ��Ԫ�ر��
                line[i][digit] = column[j][digit] = block[i / 3][j / 3][digit] = true;
                board[i][j] = digit + '0' + 1;
                dfsSolveSudoku(board, pos + 1);
                // ��ǰԪ�ػ�����ɣ��л�Ϊ��һ��Ԫ�أ���ǹ�λ
                line[i][digit] = column[j][digit] = block[i / 3][j / 3][digit] = false;
            }
        }
    }

    /**
     * @brief   ������
     *  ��дһ������ͨ�����ո�������������⡣
     *
     * �����Ľⷨ�� ��ѭ���¹���
     *
     * ���� 1-9 ��ÿһ��ֻ�ܳ���һ�Ρ�
     * ���� 1-9 ��ÿһ��ֻ�ܳ���һ�Ρ�
     * ���� 1-9 ��ÿһ���Դ�ʵ�߷ָ��� 3x3 ����ֻ�ܳ���һ�Ρ�����ο�ʾ��ͼ��
     * �������ֿո��������������֣��հ׸��� '.' ��ʾ��
     *
     * ��򵥵ķ����ǣ�
     * ��һ�������¼ÿ�������Ƿ���֡�
     * �ȱ��������������飬��¼������Ԫ�س��ֵ�λ�á�
     * ֮����еݹ�ö�٣�ͨ���ж�����������Ƿ��ڵ�ǰ���У��У��Ź����г��֣������ݵݹ��Ψһ�Ľ⡣
     *
     * @param board �������Ҫ�����������֤������������һ����
     */
    void solveSudoku(vector<vector<char>> &board)
    {
        memset(line, 0, sizeof(line));
        memset(column, 0, sizeof(column));
        memset(block, 0, sizeof(block));
        valid = false;

        for (int i = 0; i < 9; ++i)
        {
            for (int j = 0; j < 9; ++j)
            {
                if (board[i][j] == '.')
                {
                    spaces.emplace_back(i, j);
                }
                else
                {
                    int digit = board[i][j] - '0' - 1;
                    line[i][digit] = column[j][digit] = block[i / 3][j / 3][digit] = true;
                }
            }
        }
        dfsSolveSudoku(board, 0);
    }

    /**
     * @brief �������
     * 1.     1
     * 2.     11
     * 3.     21
     * 4.     1211
     * 5.     111221
     * ��һ�������� 1
     * ����ǰһ�������� 1 �� �� һ �� 1 �������� "11"
     * ����ǰһ�������� 11 �� �� �� �� 1 �� ������ "21"
     * ����ǰһ�������� 21 �� �� һ �� 2 + һ �� 1 �� ������ "1211"
     * ����ǰһ�������� 1211 �� �� һ �� 1 + һ �� 2 + �� �� 1 �� ������ "111221"
     */
    string countAndSay(int n)
    {
        string prev = "1";
        for (int i = 2; i <= n; ++i)
        {
            string curr = "";
            int start = 0;
            int pos = 0;

            while ((pos < prev.size()))
            {
                /* code */
                while (pos < prev.size() && prev[pos] == prev[start])
                {
                    pos++;
                }
                curr += to_string(pos - start) + prev[start];
                start = pos;
            }
            prev = curr;
        }
        return prev;
    }

  public:
    vector<int> cur;
    /**
     * @brief ����ܺ͵Ļ����㷨
     *
     * @param begin ��ʼλ��
     * @param sum   Ŀǰ�ļ�������ܺͣ���Ϊ�����ж���������������
     * @param candidates ����
     * @param target    Ŀ���ܺ�����
     * @param res   ��Ž��������
     */
    void dfsCombinationSum(int begin, int sum, vector<int> &candidates, int target, vector<vector<int>> &res)
    {
        if (sum == target)
        {
            res.push_back(cur);
            return;
        }
        if (sum > target)
            return;

        // ��δ������ϵ�Ԫ������㣬�ų������ظ������
        for (int i = begin; i < candidates.size(); i++)
        {
            if (target - candidates[i] < 0) // ��������󣬿��Լ��ٺ�������в���
                break;
            if (i > begin && candidates[i] == candidates[i - 1]) // ȥ���ظ����
                continue;
            int rs = candidates[i] + sum;

            cur.push_back(candidates[i]); // ��������Ԫ�ؼ����¼��ϵ�������
            // dfsCombinationSum(i , rs, candidates, target, res);////candidates �е�ÿ��������ÿ���������ʹ�ö��
            dfsCombinationSum(i + 1, rs, candidates, target, res); // candidates �е�ÿ��������ÿ�������ֻ��ʹ��һ��
            cur.pop_back(); // ���������Ԫ��ȥ������Ѱ��һ�����
        }
    }
    /**
     * @brief ����ܺ�
     * ����һ�� ���ظ�Ԫ�� ���������� candidates ��һ��Ŀ������ target ���ҳ� candidates �п���ʹ���ֺ�ΪĿ���� target
     * �� ���� ��ͬ��� �������б���ʽ���ء� ����԰� ����˳�� ������Щ��ϡ� candidates �е� ͬһ�� ���ֿ���
     * �������ظ���ѡȡ ���������һ�����ֵı�ѡ������ͬ������������ǲ�ͬ�ġ� ���ڸ��������룬��֤��Ϊ target
     * �Ĳ�ͬ��������� 150 ����
     *
     * @param candidates ���ظ�Ԫ�ص���������
     * @param target    Ŀ������
     * @return vector<vector<int>> ��ΪĿ����������б�
     */
    vector<vector<int>> combinationSum(vector<int> &candidates, int target)
    {
        vector<vector<int>> res;
        sort(candidates.begin(), candidates.end());
        dfsCombinationSum(0, 0, candidates, target, res);
        return res;
    }

    /**
     * @brief ȫ���л����㷨����
     *
     * @param res ����������н��������
     * @param output �ݴ浥�����е�����
     * @param first ����������ʼλ��
     * @param len   �������е����鳤��
     */
    void dftPermute(vector<vector<int>> &res, vector<int> &output, int first, int len)
    {
        if (first == len)
        {
            res.emplace_back(output);
            return;
        }

        for (int i = first; i < len; ++i)
        {
            swap(output[i], output[first]);
            dftPermute(res, output, first + 1, len);
            swap(output[i], output[first]);
        }
    }

    /**
     * @brief ȫ����
     * ����һ�������ظ����ֵ����� nums �������� ���п��ܵ�ȫ���� ������� ������˳�� ���ش𰸡�
     * @param nums �����ظ����ֵ�����
     * @return vector<vector<int>>
     */
    vector<vector<int>> permute(vector<int> &nums)
    {
        vector<vector<int>> res;
        dftPermute(res, nums, 0, (int)nums.size());
        return res;
    }

    /**
     * @brief ȱʧ�ĵ�һ������
     *  һ��δ�������������nums���ҳ�����û�г��ֵ���С��������
     *
     * @param nums  δ���������
     * @return int  Ŀ����С������
     */
    int firstMissingPositive(vector<int> &nums)
    {
        int n = nums.size();
        // �������飬��С��0��Ԫ���滻Ϊn+1
        for (int &num : nums)
        {
            if (num <= 0)
                num = n + 1;
        }

        // ��Ԫ�ض�Ӧ����λ���ϵ�Ԫ��ȡ��ֵ��ͨ��������ǳ���Щ�������Ǵ��ڵ�
        // ����ֵ�Ƿ�ֹ���ǵ�λ��Ԫ��������ڵ�������������Ҳ�����ڷ�ֹ�ظ���ӡ�Ҳ��ֹʹ�ñ�Ǻ�����������ж�
        for (int i = 0; i < n; i++)
        {
            int num = abs(nums[i]);
            if (num <= n)
                nums[num - 1] = -abs(nums[num - 1]);
        }

        // �������飬�ҵ���һ�������������������ظ�����+1��ֻҪ����ֵ����ʾ��λ�ö�Ӧ�����������������д��ڣ���һ�����������ı�ǵ�λ�ñ�ʾ��ֵ����Ŀ��������
        for (int i = 0; i < n; ++i)
        {
            if (nums[i] > 0)
                return i + 1;
        }

        return n + 1;
    }

    /**
     * @brief ����ˮ
     *  ����n���Ǹ�������ʾÿ�����Ϊ1�����ӵĸ߶�ͼ�����㰴�����е����ӣ�����֮���ܽӶ�����ˮ��
     *
     * @param height ���Ӹ߶�����
     * @return int  �ܽӵ���ˮ��С
     */
    int trap(vector<int> &height)
    {
        int n = height.size();
        if (n == 0)
        {
            return 0;
        }

        vector<int> leftMax(n);
        leftMax[0] = height[0];
        for (int i = 1; i < n; ++i)
        {
            leftMax[i] = max(leftMax[i - 1], height[i]);
        }

        vector<int> rightMax(n);
        rightMax[n - 1] = height[n - 1];
        for (int i = n - 2; i >= 0; --i)
        {
            rightMax[i] == max(rightMax[i + 1], height[i]);
        }

        int ans = 0;
        for (int i = 0; i < n; ++i)
        {
            ans += min(leftMax[i], rightMax[i]) - height[i];
        }
        return ans;
    }

  public:
    int buyChoco(vector<int> &prices, int money)
    {
        int first = INT_MAX, second = INT_MAX;
        for (auto p : prices)
        {
            if (p < first)
            {
                second = first;
                first = p;
            }
            else if (p < second)
            {
                second = p;
            }
        }
        return money < (first + second) ? money : (money - first - second);
    }
};
