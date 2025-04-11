#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <numeric>
#include <limits>
#include <deque>
#include <set>
#include <tuple>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <array>
#include <unordered_map>
#include <iosfwd>
#include <cstddef>
#include <random>
#include <Eigen/Dense>
#include <Eigen/Sparse> // надо скачать расширение
#include <cstring>
#include <memory>
#include <cassert>
#include <stdexcept>
#include <limits>
#define f first
#define s second
#define deb cout << "666"

using namespace std;

// 14383 = max(tree['parent_id'])
constexpr int MAXN = 1896 + 2; // с запасом (всего Node в графе)
constexpr int ROWS = 716552;
constexpr int uniq_parents = 491;
array<array<int, MAXN - 4>, uniq_parents> gr; // граф, ребра направлены вниз к детям
// этот граф по-любому придется создавать, чтобы мы спускались вниз по грфу
// и обновляли веса вершин
// если не создавать, мы сможем идти только вверх 
// и не будет обучения с выбором направления
int accord[uniq_parents];
int cnt_for_gr[uniq_parents]; // 0 по умолчанию - считает кол-во детей и
// удобно добавляет детей в граф
bool visited[1892]; // при обходе обращаеся к нему (перед запуском обхода обновлять)


pair<int, int> lca_bfs(int node1, int node2) {
    deque<tuple<int, vector<int>, int>> pq; // (узел, путь, глубина)
    pq.push_back({ node1, {node1}, 0 });
    pq.push_back({ node2, {node2}, 0 });

    int depth1 = -1;
    int depth2 = -1;
    vector<int> path2; // Второй путь

    while (!pq.empty()) {
        int node_id;
        vector<int> path;
        path.reserve(10);
        int depth;
        tie(node_id, path, depth) = pq.front();
        pq.pop_front();

        if (visited[node_id]) continue;
        visited[node_id] = true;

        if (node_id == node1) depth1 = depth;
        else if (node_id == node2) {
            depth2 = depth;
            path2 = path;
        }

        if (depth1 != -1 && depth2 != -1) {
            int lca_id = path[0];
            for (size_t i = 1; i < min(path.size(), path2.size()); ++i) {
                if (path[i] == path2[i]) {
                    lca_id = path[i];
                }
                else {
                    break;
                }
            }
            return { {lca_id}, abs(depth1 - depth2) };
        }

        for (int neighbor_id : gr[accord[node_id]]) {
            if (neighbor_id == -1) break; // аналог -1 для обозначения конца списка смежности
            path.push_back(neighbor_id);
            pq.push_back({ neighbor_id, path, depth + 1 });
        }
    }

    //Если не нашли, возвращаем некий некорректный результат.
    return { -1, -1 };
}


double D(int y_pred, int y_true) {
    fill(begin(visited), end(visited), false);
    pair<int, int> ans = lca_bfs(y_pred, y_true);

    if (ans.f == y_pred || ans.f == -1) {
        return 0.0;
    }
    return exp(-max(0, ans.s)); // не создаем лишнюю функцию levelDiff 
}

double hda(double n, const Eigen::MatrixXi& list_pred, const Eigen::MatrixXi& y_true) {
    if (list_pred.rows() != y_true.rows()) {
        throw std::runtime_error("Vectors must have the same size");
    }
    long double sum = 0.0;
    for (size_t i = 0; i < list_pred.rows(); ++i) {
        sum += D(list_pred(i, 1), y_true(i, 1));
    }
    return sum / n;
}

// удобный вывод vector
template<typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& vec) {
    for (size_t i = 0; i < vec.size(); ++i) {
        os << vec[i] << ' ';
    }
    os << '\n';
    return os;
}

// вывод array
template <typename T, size_t N>
std::ostream& operator<<(std::ostream& os, const std::array<T, N>& arr) {
    for (size_t i = 0; i < N; ++i) {
        os << arr[i];
    }
    return os;
}


// обучение для gr
// сначала напишем код в одном файле, потом раскидаем по папкам
// модель - дерево, в каждой вершине будет массив вероятностей классификаций
// цель: нахождение оптимальных весов для каждой вершины
// используем линейную регрессию с loss в каждой вершине
// шаг 1: написать mse и gradient

struct Model {

    struct Node {

        struct LogisticRegression {

         // у нас веса должны быть для каждого класса в cat_id
         Eigen::MatrixXd weights_;   // Веса линейной регрессии
         Eigen::MatrixXd bias_;

         LogisticRegression() = default;
         ~LogisticRegression() = default;
         LogisticRegression();

         shared_ptr<Eigen::MatrixXd> logit(const Eigen::MatrixXd& X);
         shared_ptr<Eigen::MatrixXd> softmax(shared_ptr<Eigen::MatrixXd> X);
         shared_ptr<Eigen::MatrixXd> _predict_proba_internal(const Eigen::MatrixXd& X);
         double loss_function(const Eigen::MatrixXd& y_pred, const int y_true);

        };

        int number_;
        LogisticRegression LogNode_;
        int number_parent_;
        vector<shared_ptr<Node>> children_;
        shared_ptr<Node> parent_;

        ~Node() = default;
        Node(int number, int number_parent, shared_ptr<Node> parent);

        Eigen::MatrixXd get_weights(); // не использовать при обучении
        // void addNode(int number); хз пока зачем это надо
    };

    Eigen::MatrixXd X_train_; // столбцы с label_encoding
    Eigen::MatrixXd X_test_;
    Eigen::MatrixXd y_train_;
    Eigen::MatrixXd y_test_;
    shared_ptr<Node> root; // дерево

    Eigen::MatrixXi test_pred;

    ~Model() = default;
    Model();

    void fit(int epochs = 10, double lr = 0.1);
    void build(int daddy = -1, shared_ptr<Node> root);
    shared_ptr<Node> dfs(int number = -1, shared_ptr<Node> root); // без dfs не обойтись, так как иначе нужно новую структуру данных созддавать
    void update_test_pred();
};

void Model::build(int daddy = -1, shared_ptr<Node> root) {
    visited[accord[daddy]] = true;
    for (int child : gr[accord[daddy]]) {
        if (!visited[child]) {
            root->children_.push_back(make_shared<Node>(child, daddy, root));
            build(child, root->children_.back());
        }
    }
    root->children_.push_back(make_shared<Node>(root, daddy, root));
}

shared_ptr<Model::Node> Model::dfs(const int number = -1, shared_ptr<Model::Node> root) { // перед запуском не забыть обновить visited
    visited[accord[root->number_]] = true;
    for (auto c : root->children_) {
        if (!visited[accord[c->number_]]) {
            if (c->number_ == number) { // нашли Себека!
                return c;
            }
            shared_ptr<Node> ans = dfs(number, root->children_.back());
            if (ans != nullptr) {
                return ans;
            }
        }
    }
    return nullptr;
}

Model::Model() : X_train_(573242, 1891 + 1), 
                 X_test_(143310, 1891 + 1), 
                 y_train_(573242, 1), 
                 y_test_(143310, 1), 
                 root(make_shared<Node>(-1, -2, nullptr)),
                 test_pred(573242, 1) {}

Model::Node::Node(int number, int number_parent, shared_ptr<Node> parent): number_(number), 
    number_parent_(number_parent), 
    parent_(parent)
{
    LogisticRegression LogNode_ = LogisticRegression();
    LogNode_.weights_ = Eigen::MatrixXd::Random(cnt_for_gr[accord[number_]] + 1, 1891 + 1); // + 1 - свободный член
    children_.reserve(cnt_for_gr[number] + 5);
}

shared_ptr<Eigen::MatrixXd> Model::Node::LogisticRegression::softmax(shared_ptr<Eigen::MatrixXd> X) {
    Eigen::MatrixXd result = X->array().exp();
    double sum = result.sum();
    if (sum == 0) {
        result.setZero(); // устанавливаем все элементы в 0 inplace
        return make_shared<Eigen::MatrixXd>(result);
    }
    result /= sum;
    return make_shared<Eigen::MatrixXd>(result);
}


shared_ptr<Eigen::MatrixXd> Model::Node::LogisticRegression::logit(const Eigen::MatrixXd& X) {
    return make_shared<Eigen::MatrixXd>(weights_ * X.transpose()); // проверена размерность 
}

shared_ptr<Eigen::MatrixXd> Model::Node::LogisticRegression::_predict_proba_internal(const Eigen::MatrixXd& X) {
    return softmax(logit(X));
}

Eigen::MatrixXd Model::Node::get_weights() { // не пользоваться, памяти много занимает (копирует)
    return LogNode_.weights_;
}

double Model::Node::LogisticRegression::loss_function(const Eigen::MatrixXd& y_pred, const int y_true) {
    if (y_true < 0) {
        throw std::invalid_argument("Number of rows in probabilities and size of true_labels must match.");
    }
    if (y_pred.rows() == 0) {
        return 0; //Обработка пустого вектора
    }

    double loss = 0.0;
    double prob = max(std::numeric_limits<double>::epsilon(), y_pred(y_true));
    double regularization = 0.5 * weights_.squaredNorm(); //L2-регуляризация
    return -std::log(prob) + regularization;
}

Eigen::MatrixXd& gradientDescent_for_Node(shared_ptr<Eigen::MatrixXd> y_pred, const int y_true) {
    static Eigen::MatrixXd result; // оптимизация (память выделяется только в одном месте для всех использований функции)
    if (y_pred->rows() == 0) {
        // только в листах
        result = Eigen::MatrixXd::Ones(1, 1);
        return result;
    }

    result = Eigen::MatrixXd::Zero(y_pred->rows(), 1); // инициализируем нулями (ты по жизни тоже ноль)

    if (y_true < 0) {
        throw std::out_of_range("y_true contains an index out of range.");
    }
    double prob = (*y_pred)(y_true);
    prob = std::max(prob, std::numeric_limits<double>::epsilon());
    result(y_true) = -1.0 / prob;

    return result;
}

void Model::fit(int epochs = 10, double lr = 0.1) { // обучение
    if (y_train_.rows() != X_train_.rows()) {
        throw std::invalid_argument("dimensions do not match. ");
    }
    if (y_train_.cols() != 1 || X_train_.rows() < 1 || X_train_.cols() < 1) {
        throw std::invalid_argument("dimensions do not match. ");
    }

    vector<double> losses;
    losses.reserve(epochs * 2 + 1);

    for (int q = 0; q < epochs; q++) {
        for (int i = 0; i < 573242; ++i) {
            // строим путь
            vector<shared_ptr<Node>> road_to_root;
            road_to_root.reserve(1000); // ПОСЧИТАТЬ ЗАРАНЕЕ MAX ГЛУБИНУ !!! НЕ ЗАБЫТЬ
            fill(begin(visited), end(visited), false);
            shared_ptr<Node> cur_node = dfs(y_train_(i, 1), root);
            while (cur_node->number_ != -1) {
                road_to_root.push_back(cur_node);
                cur_node = cur_node->parent_;
            }
            int lenWay = road_to_root.size();
            for (size_t j = lenWay - 1; j >= 0; --j) {
                int y_true = 0;
                if (j == 0) {
                    y_true = road_to_root[j]->number_; // дошли
                }
                else {
                    y_true = road_to_root[j - 1]->number_;
                }

                shared_ptr<Eigen::MatrixXd> y_pred = road_to_root[j]->LogNode_._predict_proba_internal(X_train_.block(i, 0, 1, 1892));
                // y_pred - одномерный массив вероятностей
                double loss = road_to_root[j]->LogNode_.loss_function(*y_pred, y_true);
                losses.push_back(loss);

                for (int i = 0; i < 1892; i++) {
                    Eigen::MatrixXd get_grad = gradientDescent_for_Node(y_pred, road_to_root[j - 1]->number_);
                    for (int j = 0; j < cnt_for_gr[accord[road_to_root[j]->number_]] + 1; ++j) {
                        road_to_root[j]->LogNode_.weights_(i, j) -= lr * get_grad(j, 1);
                    }
                }
            }
        }
        // после каждой эпохи считаем метрику hda
        // тестируем 
        update_test_pred();
        cout << hda(143310, test_pred, y_test_) << endl;
    }
}

void Model::update_test_pred() {
    for (int i = 0; i < 143310; ++i) {
        // строим путь
        while (true) { // опасно так делать, но мы рисковые и хотим спать
            shared_ptr<Eigen::MatrixXd> y_pred = root->LogNode_._predict_proba_internal(X_test_.block(i, 0, 1, 1892));
            Eigen::Index max_row, max_col;
            y_pred->maxCoeff(&max_row, &max_col);
            if (root->number_ == root->children_[max_row]->number_) {
                // мы нашли предсказание
                test_pred(i, 1) = root->number_;
                break;
            }
            root = root->children_[max_row];
        }
    }
}

// занимаемая память всей модели:
// 1896 * 1896 * 1896 - веса (самые ресурсозатратные)
// 8 000 000 000 * 8байт - выглядит терпимо (на деле хер знает)


signed main()
{
    // добавляем уникальные значения родителей
    // 491 = кол-во уникальных родителей
    ifstream parents;
    parents.open("C:/Users/sorvi/Downloads/parents (1).txt");
    int i_par = 0;
    while (parents.good()) {
        string line;
        getline(parents, line, ',');
        int len = line.size();
        for (int i = 1; i < len; i++) { line[0] = line[1]; }
        line.resize(len - 1);
        accord[i_par] = stoi(line);
        i_par++;
    }

    parents.close();

    ifstream tree;
    tree.open("C:/Users/sorvi/Downloads/category_tree.csv");

    int i = 0;
    while (tree.good()) {
        if (i == 1873) break; // эта строка последняя и пустая
        string line;
        getline(tree, line, '\n');
        vector<string> cur;
        cur.reserve(5); // в vector буфер обновляется если size * 2 >= buff
        // поэтому min(buff) = 5
        string str = "";
        int cnt_sep = 0;
        for (const char c : line) {
            if (c == ',' && cnt_sep == 0) {
                cur.push_back(str);
                str = "";
                cnt_sep++;
            }
            else if (c == ',' && cnt_sep == 1) {
                cur.push_back(str);
                break;
            } 
            else {
                str += c;
            }
        }

        if (cur[1] == "") {
            continue;
        }
        // добавляем ребро в граф
        if (cur[0] == "cat_id") continue;
        int node1 = stoi(cur[0]);
        int node2 = static_cast<int>(stod(cur[1]));
        auto it = find(begin(accord), end(accord), node2);
        size_t index = distance(begin(accord), it);
        gr[index][cnt_for_gr[index]] = node1;
        cnt_for_gr[index]++;
        i++;
    }
    //cout << unique_nodes.size(); == 1896

    tree.close();

    Model graph = Model();

    ifstream data;
    data.open("C:/Users/sorvi/Downloads/Telegram Desktop/data_train.csv");

    int i = 0;
    while (data.good()) {
        if (i == ROWS) break; // последняя и пустая строка
        string line;
        getline(data, line, ',');
        vector<string> cur;
        cur.reserve(MAXN + 1);
        int cnt_sep = 0;
        string str;
        for (char c : line) {
            if (c == ',' && cnt_sep <= 1891) {
                cur.push_back(str);
                str = "";
                cnt_sep++;
            }
            else if (c == ',' && cnt_sep == 1892) {
                cur.push_back(str);
                break;
            }
            else {
                str += c;
            }
        }

        if (cur[0] == "hash_id") continue;

        if (i < 573242) {
            for (int j = 0; j < 1891; j++) {
                graph.X_train_(i, j) = stoi(cur[j]);
            }
            graph.y_train_(i, 1) = stoi(cur.back());
        }
        else {
            for (int j = 0; j < 1891; j++) {
                graph.X_test_(i, j) = stod(cur[j]);
            }
            graph.y_test_(i, 1) = stod(cur.back());
        }

        i++;
    }

    data.close();

    // начинаем строить дерево обучения
    fill(begin(visited), end(visited), false);
    graph.build(-1, graph.root);
    graph.fit(); // обучение

    return 0;
}
