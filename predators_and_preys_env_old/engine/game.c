#include <stdlib.h>
#include <stdio.h>
#include "physics/entity.c"

typedef struct{
    entity* predators;
    entity* preys;
    int* alive;
    entity* obstacles;
    
    double x_limit;
    double y_limit;
    
    int num_preds;
    int num_preys;
    int num_obstacles;
    
    double r_obst_ub; // upper bound for obstacle radius
    double r_obst_lb; // lower bound for obstacle radius    
    double prey_radius;
    double pred_radius;
    
    double pred_speed;
    double prey_speed;
    double w_timestep; // world timestep
    int frameskip;
    int frame_count;
    
    int* prey_order;
    int* pred_order;
    
    double* preys_reward;
    double* preds_reward;
    int* prey_mask;
    int* pred_mask;
    
    double max_dist;
    double min_dist;
    int al;
} FGame;


double double_rand(){
    return (double)random() / RAND_MAX;
}

void shuffle_array(int* a, int n){
    
    for(int i=n-1; i>0; i--){
        int j = rand() % (i + 1);
      
        int temp = a[i];
        a[i] = a[j];
        a[j] = temp;
    }
}

void seed(int n){
    srand(n);
}


FGame* game_init(double xl, double yl,
                 int n_preds, int n_preys, int n_obsts,
                 double r_ub, double r_lb, double prey_r, double pred_r,
                 double pred_s, double prey_s,
                 double wt, int fskip){
                 
    FGame* F = (FGame*) malloc(sizeof(FGame));
    
    F -> predators = (entity* )malloc(sizeof(entity) * n_preds);
    F -> preys = (entity* )malloc(sizeof(entity) * n_preys);
    F -> alive = (int* )malloc(sizeof(int) * n_preys);
    F -> obstacles = (entity* )malloc(sizeof(entity) * n_obsts);
    
    F -> x_limit = xl;
    F -> y_limit = yl;
    F -> max_dist = (1 + xl) * (1 + xl) + (1 + yl) * (1 + yl);
    
    F -> num_preds = n_preds;
    F -> num_preys = n_preys;
    F -> num_obstacles = n_obsts;
    
    F -> r_obst_ub = r_ub;
    F -> r_obst_lb = r_lb;
    F -> prey_radius = prey_r;
    F -> pred_radius = pred_r;
    F -> min_dist = prey_r + pred_r;
    
    F -> pred_speed = pred_s;
    F -> prey_speed = prey_s;
    
    F -> w_timestep = wt;
    F -> frameskip = fskip;
    F -> frame_count = fskip;
    
    F -> prey_order = (int* ) malloc(sizeof(int) * n_preys);
    F -> prey_mask = (int* ) malloc(sizeof(int) * n_preys);
    for(int i=0; i<n_preys; i++){
        F -> prey_order[i] = i;
        F -> prey_mask[i] = 0;
    }
        
    F -> pred_order = (int* ) malloc(sizeof(int) * n_preds);
    F -> pred_mask = (int* ) malloc(sizeof(int) * n_preds);
    for(int i=0; i<n_preds; i++){
        F -> pred_order[i] = i;
        F -> pred_mask[i] = 0;
    }

    F -> preys_reward = (double* ) malloc(sizeof(double) * n_preys);        
    F -> preds_reward = (double* ) malloc(sizeof(double) * n_preds);
    
    F -> al = n_preys;
    return F;
}


entity get_prey(FGame* F, int i){
    return F -> preys[i];
}


int get_alive(FGame* F, int i){
    return F -> alive[i];
}


entity get_predator(FGame* F, int i){
    return F -> predators[i];
}


entity get_obstacle(FGame* F, int i){
    return F -> obstacles[i];
}


void step(FGame* F, double* action_preys, double* action_predators){
    
    FGame G = *F;
    
    for(int i=0; i<G.num_preys; i++){
        if (G.alive[i])
            move(&G.preys[i], action_preys[i], G.w_timestep);
    }
    
    for(int i=0; i<G.num_preds; i++)
        move(&G.predators[i], action_predators[i], G.w_timestep);
        
    int corrected = 1;
    int it_num = 0;
    int shuffle_count = 0;
    
    // while (corrected){
    for(int i=0; i<20; i++){
        corrected = 0;
        for(int k=0; k<G.num_preys; k++){
            int i = G.prey_order[k];
            int this_corrected = 0;
            force_clip_position(&G.preys[i], -G.x_limit, -G.y_limit, G.x_limit, G.y_limit);
            for(int j=0; j<G.num_obstacles; j++)
                this_corrected += force_not_intersect(&G.preys[i], &G.obstacles[j]);
                if (this_corrected){
                    break;
                }
        
            if (!this_corrected){
                for(int t=0; t<G.num_preys; t++){
                    int j = G.prey_order[t];
                    if (i==j)
                        continue;
                    this_corrected += force_not_intersect(&G.preys[i], &G.preys[j]);
                    if (this_corrected){
                        break;
                    }
                }
            }
            corrected += this_corrected;
            force_clip_position(&G.preys[i], -G.x_limit, -G.y_limit, G.x_limit, G.y_limit);
        }
        
        if (!corrected)
            break;
            
        if (it_num > G.num_preys * G.num_preys){
            it_num = 0;
            shuffle_array(G.prey_order, G.num_preys);
            shuffle_count += 1;
        }
        
        if (shuffle_count > G.num_preys * G.num_preys * 3)
           corrected = 0;
           
        it_num += 1;
    }
    
    corrected = 1;
    it_num = 0;
    shuffle_count = 0;
    //while (corrected){
    for(int i=0; i<20; i++){
        corrected = 0;
        for(int k=0; k<G.num_preds; k++){
            int i = G.pred_order[k];
            int this_corrected = 0;
            force_clip_position(&G.predators[i], -G.x_limit, -G.y_limit, G.x_limit, G.y_limit);
            for(int j=0; j<G.num_obstacles; j++)
                this_corrected += force_not_intersect(&G.predators[i], &G.obstacles[j]);
                if (this_corrected){
                    break;
                }
        
            if (!this_corrected){
                for(int t=0; t<G.num_preds; t++){
                    int j = G.pred_order[t];
                    if (i==j)
                        continue;
                    this_corrected += force_not_intersect(&G.predators[i], &G.predators[j]);
                    if (this_corrected) {
                        break;
                    }
                }
            }
            corrected += this_corrected;
            force_clip_position(&G.predators[i], -G.x_limit, -G.y_limit, G.x_limit, G.y_limit);
        }
 
        if (!corrected)
            break;
 
        if (it_num > G.num_preds * G.num_preds){
            it_num = 0;
            shuffle_array(G.pred_order, G.num_preds);
            shuffle_count += 1;
        }
 
        if (shuffle_count > G.num_preds * G.num_preds * 3)
           corrected = 0;
 
        it_num += 1;
    }
    
    for(int i=0; i<G.num_preys; i++){
        int flag = 0; 
        for(int j=0; j<G.num_preds; j++){
            if (G.alive[i] && is_intersect(&G.predators[j], &G.preys[i])){
               G.pred_mask[j]++;
               flag++;
               F -> al--;
            }
        }
        if (flag){
            G.alive[i] = 0;
            G.prey_mask[i]++;
        }
    }
    F -> frame_count--;
    
    // Rewarding
    if (F -> frame_count == 0){ 
        F -> frame_count = G.frameskip;
        
        if (F -> al){
            for(int i=0; i<G.num_preds; i++)
                G.preds_reward[i] = G.max_dist;
        }
        else{
            for(int i=0; i<G.num_preds; i++)
                G.preds_reward[i] *= (-10);
        }
         
        
        for(int i=0; i<G.num_preys; i++){
            if (!G.alive[i]){
                G.preys_reward[i] = 0;
                continue;
            }
            
            G.preys_reward[i] = center_distance(&G.predators[0], &G.preys[i]);
            if (G.preys_reward[i] < G.preds_reward[0])
                G.preds_reward[0] = G.preys_reward[i];
             
            for(int j=1; j<G.num_preds; j++){
                double d = center_distance(&G.predators[j], &G.preys[i]);
                if  (d < G.preys_reward[i])
                    G.preys_reward[i] =  d;
                if  (d < G.preds_reward[j])
                    G.preds_reward[j] =  d;
            }
        }
        
        for(int i=0; i<G.num_preys; i++){
            if (G.prey_mask[i]){
                G.preys_reward[i] = -100;
                G.prey_mask[i] = 0;
            }
            G.preys_reward[i] *= 0.1;
        }
            
        for(int j=0; j<G.num_preds; j++){
            if (G.pred_mask[j]){
                G.preds_reward[j] = -100 * G.pred_mask[j];
                G.pred_mask[j] = 0;
            }
            G.preds_reward[j] *= (-0.1);
        }
    }  
}

void reset(FGame* F){
    
    free(F -> obstacles);
    F -> obstacles = (entity*) malloc(sizeof(entity) * (F -> num_obstacles));
    
    free(F -> preys);
    free(F -> alive);
    F -> preys = (entity*) malloc(sizeof(entity) * (F -> num_preys));
    F -> alive = (int*) malloc(sizeof(int) * (F -> num_preys));
    
    free(F -> predators);
    F -> predators = (entity*) malloc(sizeof(entity) * (F -> num_preds));
    
    F -> al = F -> num_preys;
    
    FGame G = *F;

    for(int i=0; i<G.num_preys; i++){
            G.prey_mask[i] = 0;
            G.preys_reward[i] = 0;
    }
    
    for(int i=0; i<G.num_preds; i++){
        G.pred_mask[i] = 0;
        G.preds_reward[i] = 0;
    }
    
    for(int i=0; i<G.num_obstacles; i++){
        double r = double_rand() * (G.r_obst_ub - G.r_obst_lb) + G.r_obst_lb;
        double x = (2 * double_rand() - 1) * (G.x_limit - r);
        double y = (2 * double_rand() - 1) * (G.y_limit - r);
        entity* e = Entity_init(r, 0., x, y);
        G.obstacles[i] = *e;
        free(e);
    }
    
    for(int i=0; i<G.num_preys; i++){
        int created = 0;
        while (!created){
            created = 1;
            double x = (2 * double_rand() - 1) * (G.x_limit - G.prey_radius);
            double y = (2 * double_rand() - 1) * (G.y_limit - G.prey_radius);
            entity* e = Entity_init(G.prey_radius, G.prey_speed, x, y);
            for(int j=0; j<G.num_obstacles; j++){
                if (is_intersect(&(G.obstacles[j]), e)){
                    created = 0;
                    free(e);
                    break;
                }
            }
            if (created){
                for(int j=0; j<i; j++){
                    if (is_intersect(&(G.preys[j]), e)){
                        created = 0;
                        free(e);
                        break;
                    }
                }
            }
            if (created){
                G.preys[i] = *e;
                G.alive[i] = 1;
                free(e);
            } 
        }
    }
    
    
    for(int i=0; i<G.num_preds; i++){
        int created = 0;
        while (!created){
            created = 1;
            double x = (2 * double_rand() - 1) * (G.x_limit - G.pred_radius);
            double y = (2 * double_rand() - 1) * (G.y_limit - G.pred_radius);
            entity* e = Entity_init(G.pred_radius, G.pred_speed, x, y);
            for(int j=0; j<G.num_obstacles; j++){
                if (is_intersect(&(G.obstacles[j]), e)){
                    created = 0;
                    free(e);
                    break;
                }
            }
            if (created){
                for(int j=0; j<G.num_preys; j++){
                    if (is_intersect(&(G.preys[j]), e)){
                        created = 0;
                        free(e);
                        break;
                    }
                }
            }
            if (created){
                for(int j=0; j<i; j++){
                    if (is_intersect(&(G.predators[j]), e)){
                        created = 0;
                        free(e);
                        break;
                    }
                }
            }
            if (created){
                G.predators[i] = *e;
                free(e);
            }
        }
    }
}
