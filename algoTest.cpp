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
    // 链表的中间节点
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

    // 反转链表
    ListNode *reverseList(ListNode *head)
    { // 迭代
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
    { // 递归
        if (!head || !head->next)
            return head;

        ListNode *newHead = reverseListEx(head->next);
        head->next->next = head;
        head->next = nullptr;
        return newHead;
    }

    // 合并链表,交叉合并
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

    // 重排链表，用线性表存储，利用下标来重新排列
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

    // 重排链表，根据中点拆分为两部分，后半部分反转，交叉合并两链表
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
     *  快速选择算法，
     * 在分解的过程当中，我们会对子数组进行划分，如果划分得到的 q 正好就是我们需要的下标，就直接返回 a[q]；
     * 否则，如果 q 比目标下标小，就递归右子区间，否则递归左子区间。
     * 这样就可以把原来递归两个区间变成只递归一个区间，提高了时间效率。
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

    // 数组中的第K个最大元素
    int findKthLargest(vector<int> &nums, int k)
    {
        int n = nums.size();
        return quickselect(nums, 0, n - 1, n - k);
    }

    // 移动片段得到字符串
    /****
     * 根据相对位置分析，要想start能变成target，需要满足以下三个条件：
     * 去除下划线之后，两个串应该是相等的；
     * start中的L与target中的L是一一对应的，而L不能向右移动，所以target中第i个L绝对不能在start中第i个L的右边；
     * start中的R与target中的R是一一对应的，而R不能向左移动，所以target中第i个R绝对不能在start中第i个R的左边
     * L必须在R的左边，左右两边的L和R的数量要相等
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

        // 判断是否还有未处理的‘L’或者‘R'字符，有的话返回false
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

    // 打家劫舍,动态规划
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
            dp[k] = max(dp[k - 1], nums[k - 1] + dp[k - 2]); // 可优化
        }

        return dp[N];
    }
    // 打家劫舍,动态规划,空间优化，实际上每次只用到f(n-1)和f(n-2)的结果之前的结果已经用不到了，就不用存储了。
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

    // 到最近的人的最大距离,不断求连续的0的长度
    int maxDistToColsest(vector<int> &seats)
    {
        int res = 0;
        int l = 0;
        // 以0开始到1结尾
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
                res = max(res, r - l - 1); // 以0结尾往前到1
            else
                res = max(res, (r - l) / 2); // 两个1之间的最中间的0的位置
            */

            res = (r == seats.size()) ? max(res, r - l - 1) : max(res, (r - l) / 2);
            l = r;
        }
        return res;
    }

    // 到最近的人的最大距离,座位中里最近有人座位得到最大的空位距离
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
                    d = max(d, i - last); // 记录两个1中间的0的最大长度
                }
                if (first == -1)
                    first = i; // 记录0开始1结束为左侧长度
                last = i;
            }
        }
        // max(first, n - last - 1)//最右侧的1开始0结尾的长度
        return max({d / 2, max(first, n - last - 1)});
    }

    // 找出转圈游戏输家
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

    // 最长递增子序列,动态规划,细分为记录到每个位置的子序列大小，最后输出最大的一个子序列值
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
     * 最长递增子序列:贪心+二分查找
     *  设当前已求出的最长上升子序列的长度为 len（初始时为 1），从前往后遍历数组nums，在遍历到nums[i] 时：
     * 如果nums[i]>d[len] ，则直接加入到 d 数组末尾，并更新 len=len+1；
     * 否则，在 d 数组中二分查找，找到第一个比 nums[i] 小的数 d[k] ，并更新 d[k+1]=nums[i]。
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

    // 多数元素，指在数组中出现次数大于[n/2]的元素,哈希表
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

    // 多数元数，排序后多数元素一定会包括排序后的中间一个元素，因为多数元素次数大于[n/2]
    int majorttyElementEX(vector<int> &nums)
    {
        sort(nums.begin(), nums.end());
        return nums[nums.size() / 2];
    }

    /**********
     *  多数元素，Boyer-Moore投票算法。
     * 在每一轮投票过程中，从数组中删除两个不同的元素，直到投票过程无法继续，此时数组为空或者数组中剩下的元素都相等。
     * 因为多数元素在数组中占有超过一半，所以如果每次去除两个不同的元素，直到没有不同的元素。
     * 此时数组为空或者全为相等元素，为空表示不存在多数元素，剩下的元素都相等表示这个元素就是多数元素
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

    //   随机位置快速排序
    vector<int> sortArray(vector<int> &nums)
    {
        srand((unsigned)time(NULL));
        randomized_quicksort(nums, 0, (int)nums.size() - 1);
        return nums;
    }

    //   归并排序
    vector<int> sortArrayEX(vector<int> &nums)
    {
        tmp.resize((int)nums.size(), 0);
        mergeSort(nums, 0, (int)nums.size() - 1);
        return nums;
    }

    //   最长连续序列
    int longestConsecutive(vector<int> &nums)
    {
        // 创造无序set容器存放nums元素
        unordered_set<int> num_set;
        for (const int &num : nums)
        {
            num_set.insert(num);
        }

        int longestStreak = 0;

        for (const int &num : num_set)
        {
            // 存在前一个连续数，表示已经记录过连续数，就没必要再进行一次计数
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

    // 合并区间
    vector<vector<int>> merge(vector<vector<int>> &intervals)
    {
        if (intervals.size() == 0)
            return {};

        sort(intervals.begin(), intervals.end());
        vector<vector<int>> merged; // 定义一个二维向量

        for (int i = 0; i < intervals.size(); i++)
        {
            int left = intervals[i][0];
            int right = intervals[i][1];

            if (!merged.size() || merged.back()[1] < left)
            {
                merged.push_back({left, right});
                // 存储区间，或者后面区间与前面不相容
            }
            else
            {
                merged.back()[1] = max(merged.back()[1], right);
                // 更新最后一个区间的结束位置，条件满足最后一个区间的结束位置在后一个判断区间的起始位置之后
            }
        }
        return merged;
    }

    //  下降路径最小和，遍历计算出所有结果，返回最小结果
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
                    mn = min(mn, dp[i - 1][j - 1]); // 防止提取数据越过左侧
                if (j < n - 1)
                    mn = min(mn, dp[i - 1][j + 1]); // 防止右侧数组越界

                dp[i][j] = mn + matrix[i][j]; // 上方元素+下方三个可跳转元素中最小的一个
            }
        }

        return *min_element(dp[n - 1].begin(), dp[n - 1].end()); // 所有计算结果中最小的一个
    }

    // 跳跃游戏II
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
                // 能跳到的最远的距离
                maxPos = max(maxPos, i + nums[i]);
            }
            start = end;      // 下一次起跳点范围开始的格子
            end = maxPos + 1; // 下一次起跳点范围结束的格子
            ans++;            // 跳跃次数
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
     * 1,设 s 长度为 n，创建一个长为 n 的 replace 列表。
     * 2,遍历每个替换操作。对于第 i 个替换操作，如果从indices[i] 开始的字符串有前缀 sources[i]，则可以替换成 target[i]。
     *   例如s="abcd"，s[1:]="bcd" 有前缀 "bc"。
     *   此时记录replace[indices[i]]=(target[i],len(sources[i]))，表示替换后的字符串，以及被替换的长度。
     *
     * 3,初始化 i=0，如果 replace[i] 是空的，那么无需替换，把 s[i]加入答案，然后 i 加一；
     *  如果 replace[i] 不为空，那么把replace[i][0] 加入答案，然后 i 增加replace[i][1]。循环直到 i=n 为止。
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

    //   旋转数组，使用额外数组来存储数据
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

    // 翻转函数
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
     * 我们可以先将所有元素翻转，这样尾部的 k?mod?n个元素就被移至数组头部，然后我们再翻转 [0,k?mod?n?1]区间的元素和
     * [k?mod?n,n?1]区间的元素即能得到最后的答案。
     * */
    void rotateEx1(vector<int> &nums, int k)
    {
        k %= nums.size();
        reverse(nums, 0, nums.size() - 1);
        reverse(nums, 0, k - 1);
        reverse(nums, k, nums.size() - 1);
    }

    /********
     * 翻转卡片游戏
     * 如果一张卡片正反两面有相同的数字，那么这张卡片无论怎么翻转，正面都是这个数字，这个数字即不能是最后所选的数字 x。
     * 按照这个思路，我们首先遍历所有卡片，如果卡片上的两个数字相同，则加入哈希集合 same
     * 中，除此集合外的所有数字，都可以被选做 x，我们只需要再次遍历所有数字，找到最小值即可。
     * 最后，我们返回找到的最小值，如果没有则返回 0。
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
     * @brief 有效的数独
     *  请你判断一个 9 x 9 的数独是否有效。只需要 根据以下规则 ，验证已经填入的数字是否有效即可。
     * 数字 1-9 在每一行只能出现一次。
     * 数字 1-9 在每一列只能出现一次。
     * 数字 1-9 在每一个以粗实线分隔的 3x3 宫内只能出现一次。
     *
     * @param board  需要判断的目标数独
     * @return true 传入的数独已经填入的数字有效
     * @return false 传入的数独已经填入的数字无效
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
    bool line[9][9];     // 行
    bool column[9][9];   // 列
    bool block[3][3][9]; // 九宫格
    bool valid;
    vector<pair<int, int>> spaces; // 用于存放数独元素的坐标

  public:
    /**
     * @brief 解数独的唯一解的回溯函数
     * 通过判断填入数组的当前行，列和九宫格是否出现，来递归回溯出唯一解
     *
     * @param board 目标数独
     * @param pos 填入新数字的位置
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
            if (!line[i][digit] && !column[j][digit] && !block[i / 3][j / 3][digit]) // 判断填入的数字是否有效
            {
                // 进入下一层填数，将当前位置元素标记
                line[i][digit] = column[j][digit] = block[i / 3][j / 3][digit] = true;
                board[i][j] = digit + '0' + 1;
                dfsSolveSudoku(board, pos + 1);
                // 当前元素回溯完成，切换为下一个元素，标记归位
                line[i][digit] = column[j][digit] = block[i / 3][j / 3][digit] = false;
            }
        }
    }

    /**
     * @brief   解数独
     *  编写一个程序，通过填充空格来解决数独问题。
     *
     * 数独的解法需 遵循如下规则：
     *
     * 数字 1-9 在每一行只能出现一次。
     * 数字 1-9 在每一列只能出现一次。
     * 数字 1-9 在每一个以粗实线分隔的 3x3 宫内只能出现一次。（请参考示例图）
     * 数独部分空格内已填入了数字，空白格用 '.' 表示。
     *
     * 最简单的方法是：
     * 用一个数组记录每个数字是否出现。
     * 先遍历整个数独数组，记录数独中元素出现的位置。
     * 之后进行递归枚举，通过判断填入的数字是否在当前的行，列，九宫格中出现，来回溯递归出唯一的解。
     *
     * @param board 传入的需要解的数独，保证输入数独仅有一个解
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
     * @brief 外观数列
     * 1.     1
     * 2.     11
     * 3.     21
     * 4.     1211
     * 5.     111221
     * 第一项是数字 1
     * 描述前一项，这个数是 1 即 “ 一 个 1 ”，记作 "11"
     * 描述前一项，这个数是 11 即 “ 二 个 1 ” ，记作 "21"
     * 描述前一项，这个数是 21 即 “ 一 个 2 + 一 个 1 ” ，记作 "1211"
     * 描述前一项，这个数是 1211 即 “ 一 个 1 + 一 个 2 + 二 个 1 ” ，记作 "111221"
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
     * @brief 组合总和的回溯算法
     *
     * @param begin 起始位置
     * @param sum   目前的计算出的总和，作为跳出判断条件，结束回溯
     * @param candidates 数组
     * @param target    目标总和整数
     * @param res   存放结果的数组
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

        // 以未参与组合的元素做起点，排除出现重复的组合
        for (int i = begin; i < candidates.size(); i++)
        {
            if (target - candidates[i] < 0) // 数组排序后，可以减少后面的运行步骤
                break;
            if (i > begin && candidates[i] == candidates[i - 1]) // 去除重复结果
                continue;
            int rs = candidates[i] + sum;

            cur.push_back(candidates[i]); // 将数组中元素加入记录组合的数组中
            // dfsCombinationSum(i , rs, candidates, target, res);////candidates 中的每个数字在每个组合中能使用多次
            dfsCombinationSum(i + 1, rs, candidates, target, res); // candidates 中的每个数字在每个组合中只能使用一次
            cur.pop_back(); // 将最后加入的元素去除，搜寻下一个组合
        }
    }
    /**
     * @brief 组合总和
     * 给你一个 无重复元素 的整数数组 candidates 和一个目标整数 target ，找出 candidates 中可以使数字和为目标数 target
     * 的 所有 不同组合 ，并以列表形式返回。 你可以按 任意顺序 返回这些组合。 candidates 中的 同一个 数字可以
     * 无限制重复被选取 。如果至少一个数字的被选数量不同，则两种组合是不同的。 对于给定的输入，保证和为 target
     * 的不同组合数少于 150 个。
     *
     * @param candidates 无重复元素的整数数组
     * @param target    目标整数
     * @return vector<vector<int>> 和为目标数的组合列表
     */
    vector<vector<int>> combinationSum(vector<int> &candidates, int target)
    {
        vector<vector<int>> res;
        sort(candidates.begin(), candidates.end());
        dfsCombinationSum(0, 0, candidates, target, res);
        return res;
    }

    /**
     * @brief 全排列回溯算法函数
     *
     * @param res 存放所有排列结果的数组
     * @param output 暂存单次排列的数组
     * @param first 数组排列起始位置
     * @param len   用于排列的数组长度
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
     * @brief 全排列
     * 给定一个不含重复数字的数组 nums ，返回其 所有可能的全排列 。你可以 按任意顺序 返回答案。
     * @param nums 不含重复数字的数组
     * @return vector<vector<int>>
     */
    vector<vector<int>> permute(vector<int> &nums)
    {
        vector<vector<int>> res;
        dftPermute(res, nums, 0, (int)nums.size());
        return res;
    }

    /**
     * @brief 缺失的第一个正数
     *  一个未排序的整数数组nums，找出其中没有出现的最小的正整数
     *
     * @param nums  未排序的整数
     * @return int  目标最小正整数
     */
    int firstMissingPositive(vector<int> &nums)
    {
        int n = nums.size();
        // 遍历数组，将小于0的元素替换为n+1
        for (int &num : nums)
        {
            if (num <= 0)
                num = n + 1;
        }

        // 将元素对应索引位置上的元素取负值，通过这样标记出哪些正整数是存在的
        // 绝对值是防止打标记的位置元素满足存在的正整数条件，也是用于防止重复添加。也防止使用标记后的数据来做判断
        for (int i = 0; i < n; i++)
        {
            int num = abs(nums[i]);
            if (num <= n)
                nums[num - 1] = -abs(nums[num - 1]);
        }

        // 遍历数组，找到第一个正整数的索引并返回该索引+1，只要是正值，表示该位置对应的正整数不在数组中存在，第一个满足条件的标记的位置表示的值就是目标正整数
        for (int i = 0; i < n; ++i)
        {
            if (nums[i] > 0)
                return i + 1;
        }

        return n + 1;
    }

    /**
     * @brief 接雨水
     *  给定n个非负整数表示每个宽度为1的柱子的高度图，计算按此排列的柱子，下雨之后能接多少雨水。
     *
     * @param height 柱子高度数组
     * @return int  能接的雨水大小
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
