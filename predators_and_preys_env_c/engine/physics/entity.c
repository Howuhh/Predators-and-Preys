#include <stdlib.h>
#include <math.h>
#include <stdio.h>

typedef struct{
    double radius;
    double speed;
    double position[2];
} entity;

entity * Entity_init(double r, double s, double x, double y){

    entity* Ent = malloc(sizeof(entity));
    
    Ent -> position[0] = x;
    Ent -> position[1] = y;
    Ent -> radius = r;
    Ent -> speed = s;
    
    return Ent;
} 

void Entity_delete(entity* Ent){
    free(Ent);
}

void move(entity* Ent, double angle, double timestep){
    double PI = 3.1415926535897932;
    Ent -> position[0] += Ent -> speed * cos(PI * angle) * timestep;
    Ent -> position[1] += Ent -> speed * sin(PI * angle) * timestep;
}

double center_distance(entity* Ent, entity* Other){
    double x = Ent -> position[0] - Other -> position[0];
    double y = Ent -> position[1] - Other -> position[1];
    return sqrt(x * x + y * y); 
}

double real_distance(entity* Ent, entity* Other){
    double x = center_distance(Ent, Other);
    double y = Ent -> radius + Other -> radius;
    return x - y; 
}

int is_intersect(entity* Ent, entity* Other){
    return center_distance(Ent, Other) < (Ent -> radius + Other -> radius) ? 1 : 0;
}


int force_not_intersect(entity* Ent, entity* Other){
    if (is_intersect(Ent, Other)){
        double v_x = Ent -> position[0] - Other -> position[0];
        double v_y = Ent -> position[1] - Other -> position[1];
        double norm = sqrt(v_x * v_x + v_y * v_y) + 1e-8;
        v_x /= norm;
        v_y /= norm;
        double r = (Ent -> radius + Other -> radius) * (1 + 1e-2);
        v_x *= r;
        v_y *= r;
        Ent -> position[0] = Other -> position[0] + v_x;
        Ent -> position[1] = Other -> position[1] + v_y;
        return 1;
    }
    else
        return 0;
}
      
void force_clip_position(entity* Ent, double min_x, double min_y, double max_x, double max_y){

    double r = Ent -> radius;
    
    if (Ent -> position[0] < min_x + r + 1e-2)
        Ent -> position[0] = min_x + r + 1e-2;
    else if (Ent -> position[0] > max_x - r - 1e-2)
        Ent -> position[0] = max_x - r - 1e-2;
        
    if (Ent -> position[1] < min_y + r + 1e-2)
        Ent -> position[1] = min_y + r + 1e-2;
    else if (Ent -> position[1] > max_y - r - 1e-2)
        Ent -> position[1] = max_y - r - 1e-2;
}
