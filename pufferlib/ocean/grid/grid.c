#include "grid.h"

/*
unsigned int actions[41] = {NORTH, NORTH, NORTH, NORTH, NORTH, NORTH,
    EAST, EAST, EAST, EAST, EAST, EAST, SOUTH, WEST, WEST, WEST, NORTH, WEST,
    WEST, WEST, SOUTH, SOUTH, SOUTH, SOUTH, SOUTH, SOUTH, SOUTH, SOUTH, SOUTH,
    SOUTH, SOUTH, SOUTH, EAST, EAST, EAST, EAST, EAST, EAST, EAST, EAST, SOUTH
};

void test_multiple_envs() {
    Env** envs = (Env**)calloc(10, sizeof(Env*));
    for (int i = 0; i < 10; i++) {
        envs[i] = alloc_locked_room_env();
        reset_locked_room(envs[i]);
    }

    for (int i = 0; i < 41; i++) {
        for (int j = 0; j < 10; j++) {
            envs[j]->actions[0] = actions[i];
            step(envs[j]);
        }
    }
    for (int i = 0; i < 10; i++) {
        free_allocated_grid(envs[i]);
    }
    free(envs);
    printf("Done\n");
}

int main() {
    int width = 32;
    int height = 32;
    int num_agents = 1;
    int horizon = 128;
    float agent_speed = 1;
    int vision = 5;
    bool discretize = true;

    int render_cell_size = 32;
    int seed = 42;

    //test_multiple_envs();
    //exit(0);

    Env* env = alloc_locked_room_env();
    reset_locked_room(env);

    Renderer* renderer = init_renderer(render_cell_size, width, height);

    int t = 0;
    while (!WindowShouldClose()) {
        // User can take control of the first agent
        env->actions[0] = PASS;
        if (IsKeyDown(KEY_UP)    || IsKeyDown(KEY_W)) env->actions[0] = NORTH;
        if (IsKeyDown(KEY_DOWN)  || IsKeyDown(KEY_S)) env->actions[0] = SOUTH;
        if (IsKeyDown(KEY_LEFT)  || IsKeyDown(KEY_A)) env->actions[0] = WEST;
        if (IsKeyDown(KEY_RIGHT) || IsKeyDown(KEY_D)) env->actions[0] = EAST;

        //for (int i = 0; i < num_agents; i++) {
        //    env->actions[i] = rand() % 4;
        //}
        //env->actions[0] = actions[t];
        bool done = step(env);
        if (done) {
            printf("Done\n");
            reset_locked_room(env);
        }
        render_global(renderer, env);

    }
    close_renderer(renderer);
    free_allocated_grid(env);
    return 0;
}
*/

int main() {
    int max_size = 32;
    int width = 32;
    int height = 32;
    int num_agents = 1;
    int horizon = 128;
    float speed = 1;
    int vision = 5;
    bool discretize = true;

    int render_cell_size = 32;
    int seed = 42;

    Env* env = allocate_grid(max_size, num_agents, horizon,
        vision, speed, discretize);


    env->width = 32;
    env->height = 32;
    env->agents[0].spawn_x = 16;
    env->agents[0].spawn_y = 16;
    env->agents[0].color = 6;
    //reset(env, seed);
    //load_locked_room_preset(env);
 

    width = height = 31;
    env->width=31;
    env->height=31;
    env->agents[0].spawn_x = 1;
    env->agents[0].spawn_y = 1;
    reset(env, seed);
    generate_growing_tree_maze(env->grid, env->width, env->height, max_size, 0.85, 0);
    env->grid[(env->height-2)*env->max_size + (env->width - 2)] = GOAL;
 
    Renderer* renderer = init_renderer(render_cell_size, width, height);

    int tick = 0;
    while (!WindowShouldClose()) {
        // User can take control of the first agent
        env->actions[0] = ATN_FORWARD;
        Agent* agent = &env->agents[0];
        // TODO: Why are up and down flipped?
        if (IsKeyDown(KEY_UP)    || IsKeyDown(KEY_W)){
            agent->direction = 3.0*PI/2.0;
        } else if (IsKeyDown(KEY_DOWN)  || IsKeyDown(KEY_S)) {
            agent->direction = PI/2.0;
        } else if (IsKeyDown(KEY_LEFT)  || IsKeyDown(KEY_A)) {
            agent->direction = PI;
        } else if (IsKeyDown(KEY_RIGHT) || IsKeyDown(KEY_D)) {
            agent->direction = 0;
        } else {
            env->actions[0] = ATN_PASS;
        }

        //for (int i = 0; i < num_agents; i++) {
        //    env->actions[i] = rand() % 4;
        //}
        //env->actions[0] = actions[t];
        tick = (tick + 1)%12;
        bool done = false;
        if (tick % 12 == 0) {
            done = step(env);

        }
        if (done) {
            printf("Done\n");
            reset(env, seed);
        }
        render_global(renderer, env, (float)tick/12.0);
    }
    close_renderer(renderer);
    free_allocated_grid(env);
    return 0;
}

