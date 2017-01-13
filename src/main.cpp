#include "image.hpp"
#include "model.hpp"
#include <cstdio>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/gtc/matrix_transform.hpp>

float rotate_x = 0.f;
float rotate_y = 0.f;
glm::mat4 view;
glm::vec3 target;
float min_distance = 3.f;
float max_distance = 100.f;
float distance = 5.f;

double mouse_press_x = 0.f;
double mouse_press_y = 0.f;

void update_view()
{
    glm::vec3 offset = distance * glm::vec3{
        glm::cos(rotate_y) * glm::sin(rotate_x),
        glm::sin(rotate_y),
        glm::cos(rotate_y) * glm::cos(rotate_x)
    };
    view = glm::lookAt(
        target + offset,
        target,
        glm::vec3{0.f, 1.f, 0.f}
        );
}

void on_scroll(GLFWwindow* window, double x, double y)
{
    double speed = 0.5f;
    distance = glm::clamp(static_cast<float>(distance + -speed * y), min_distance, max_distance);
    update_view();
}

void on_cursor_pos(GLFWwindow* window, double x, double y)
{
    double speed = 0.01f;
    rotate_x = rotate_x + -speed * (x - mouse_press_x);
    rotate_y = glm::clamp(rotate_y + speed * (y - mouse_press_y), -glm::pi<double>() / 2 + 0.1f, glm::pi<double>() / 2 - 0.1f);
    mouse_press_x = x;
    mouse_press_y = y;
    update_view();
}

void on_mouse_button(GLFWwindow* window, int button, int action, int mods)
{
    if (button == 0) {
        if (action == GLFW_PRESS) {
            glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
            glfwGetCursorPos(window, &mouse_press_x, &mouse_press_y); 
            glfwSetCursorPosCallback(window, on_cursor_pos);
        } else if (action == GLFW_RELEASE) {
            glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
            glfwSetCursorPosCallback(window, nullptr);
        }
    } 
}

void make_grid(std::vector<VertPC>& grid, int size, const glm::vec3& color)
{
    float half_size = size / 2.f;
    for (int i = 0; i <= size; i++) {
        grid.push_back({glm::vec3{-half_size + i, 0, -half_size}, color});
        grid.push_back({glm::vec3{-half_size + i, 0,  half_size}, color});
        grid.push_back({glm::vec3{-half_size, 0, -half_size + i}, color});
        grid.push_back({glm::vec3{ half_size, 0, -half_size + i}, color});
    }
}

int main()
{
    if (not glfwInit()) {
        fprintf(stderr, "Failed to initialize GLFW.\n");
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);

    int window_width = 800;
    int window_height = 600;
    const char* window_title = "Model Loading";

    GLFWwindow* window = glfwCreateWindow(window_width, window_height, window_title, nullptr, nullptr);
    if (window == nullptr) {
        fprintf(stderr, "Failed to create window.\n");
        return -1;
    }
    glfwMakeContextCurrent(window);

    if (not gladLoadGLLoader(reinterpret_cast<GLADloadproc>(glfwGetProcAddress))) {
        fprintf(stderr, "Failed to initialize GLAD.\n");
        return -1;
    }

    glfwSetMouseButtonCallback(window, on_mouse_button);
    glfwSetScrollCallback(window, on_scroll);
    update_view();

    ShaderManager sm;
    ImageLoader il;
    DrawUtil du {&sm};
    ModelManager mm {&sm, &il, &du};

    if (not il.init()) {
        fprintf(stderr, "Failed to initialize image loader.\n");
        return -1;
    }

    if (not du.init()) {
        fprintf(stderr, "Failed to initialize draw util.\n");
        return -1;
    }

    if (not mm.init()) {
        fprintf(stderr, "Failed to initialize model manager.\n");
        return -1;
    }

    std::vector<VertPC> grid;
    make_grid(grid, 10, glm::vec3{0.3f, 0.3f, 0.3f});

    Model mario;
    Animation mario_walk;
    mm.analyze_model("models/mario/mario.fbx");
    mm.load_model(&mario, &mario_walk, "models/mario/mario.fbx");
    Pose pose = mario.default_pose;

    target = (mario.bbox.min + mario.bbox.max) / 2.f;
    distance = glm::length(mario.bbox.max - mario.bbox.min) / 2.f;
    min_distance = distance * 0.8f;
    max_distance = distance * 100.f;
    update_view();

    float aspect = static_cast<float>(window_width) / window_height;
    glm::mat4 projection = glm::perspective(1.f, aspect, 0.1f, 1000.f);

    glClearColor(0.5f, 0.5f, 0.5f, 0.f);

    while (not glfwWindowShouldClose(window)) {
        glfwPollEvents();
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        mm.update_pose(&mario, pose, &mario_walk, glfwGetTime());
        glEnable(GL_DEPTH_TEST);
        mm.draw_model(&mario, pose, projection, view);
        du.draw(GL_LINES, projection, view, grid);
        glDisable(GL_DEPTH_TEST);
        mm.draw_skeleton(&mario, pose, projection, view);
        glfwSwapBuffers(window);
    }

    glfwTerminate();
}
