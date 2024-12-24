#define _USE_MATH_DEFINES // for M_PI
#define NOMINMAX // so that std::min is used instead of the windows.h version
#include <windows.h> // Includes the core Windows API header file, which provides functions for creating windows, handling events, and drawing on the screen.
#include <windowsx.h>
#include <cmath>
#include <iostream>
#include <fstream>
#include <map>
#include "PRNGs.h"
#include <sycl/sycl.hpp>
#include "RayTracerSYCL.h"
#include <stack>
#include <utility>

// Create a queue using a GPU selector
sycl::queue Q{ sycl::gpu_selector_v };

bool DEBUG = false;

// faster pow() function for when using positive integer powers
double power(double x, int p) {
	double product = 1;
	for (int i = 0; i < p; i++) product *= x;
	return product;
}
// simple dot product function
double dot3D(const double* a, const double* b) {
	return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
// simple normalization function
void normalize(double* v, int dim) {
	double length = 0;
	for (int i = 0; i < dim; i++) {
		length += power(v[i], 2);
	}
	length = sqrt(length);
	for (int i = 0; i < dim; i++) {
		v[i] /= length;
	}
}
// rotates 3d vector around x-axis
void rotateX(double* v, double angle) { // angle in radians
	double temp[3] = { v[0], cos(angle) * v[1] - sin(angle) * v[2], sin(angle) * v[1] + cos(angle) * v[2] };
	v[0] = temp[0];
	v[1] = temp[1];
	v[2] = temp[2];
}
// rotates 3d vector around y-axis
void rotateY(double* v, double angle) { // angle in radians
	double temp[3] = {
		cos(angle) * v[0] + sin(angle) * v[2],
		v[1],
		-sin(angle) * v[0] + cos(angle) * v[2]
	};
	v[0] = temp[0];
	v[1] = temp[1];
	v[2] = temp[2];
}
double vectorLength3D(const double* V) {
	return sqrt(V[0] * V[0] + V[1] * V[1] + V[2] * V[2]);
}
double distance3D(const double* A, const double* B) {
	double C[3] = { A[0] - B[0], A[1] - B[1], A[2] - B[2] };
	return vectorLength3D(C);
}
double getAngle(const double* A, const double* B, const double* C) { // B is crux of angle
	double BA[3]; // vector from B to A
	double BC[3]; // vector from B to C
	for (int channel = 0; channel < 3; channel++) {
		BA[channel] = A[channel] - B[channel];
		BC[channel] = C[channel] - B[channel];
	}
	return acos(dot3D(BA, BC) / (vectorLength3D(BA) * vectorLength3D(BC)));
}
double getAngleVectors(const double* BA, const double* BC) { // B is crux of angle
	return acos(dot3D(BA, BC) / (vectorLength3D(BA) * vectorLength3D(BC)));
}

void randDirection(double* dir, uint64_t* rand_seed) {

	// from here: https://stackoverflow.com/questions/5408276/sampling-uniformly-distributed-random-points-inside-a-spherical-volume

	double phi = randDouble(rand_seed) * M_PI * 2;
	double costheta = randDouble(rand_seed) * 2 - 1;

	double theta = acos(costheta);
	dir[0] = sin(theta) * cos(phi);
	dir[1] = sin(theta) * sin(phi);
	dir[2] = cos(theta);

}

// GLOBAL VARIABLES
std::ofstream debug_file("debug_output.txt");

// create materials
class Material {
public:
	double color[3];        // Only used when sphere is emissive
	double diffuse_color[3];
	double emissivity;
	double refr_index;
	double roughness;

	// Make the constructor constexpr for compile-time evaluation
	constexpr Material(double color_r, double color_g, double color_b, double emissivity,
		double diffuse_color_r, double diffuse_color_g, double diffuse_color_b,
		double roughness, double refr_index)
		: color{ color_r, color_g, color_b },
		diffuse_color{ diffuse_color_r, diffuse_color_g, diffuse_color_b },
		emissivity(emissivity),
		refr_index(refr_index),
		roughness(roughness) {}
};

constexpr Material light(255, 255, 255, 1.0, // light
	255, 255, 255, 1.0,
	1.5);
constexpr Material wall(255, 255, 255, 0.0, // wall
	255, 0, 255, 0.0,
	1.33);
constexpr Material air(255, 255, 255, 0.0, // air
	255, 0, 255, 0.5,
	1.0);
constexpr Material light2(0, 255, 255, 1.0, // light2
	255, 255, 255, 1.0,
	1.5);

class Object {
public:

	double pos[3];
	const Material* M;
	double light_radius; // when an object is emissive, for ambient lighting it is treated as a sphere of this radius
	void* parent = nullptr; // cast to group pointer
	int precedence;
	char obj_type; // group, sphere, box, etc.

	Object() {}

	// assumes view_dir is normalized, which it should be
	virtual double getDistance(const double* view_dir, const double* view_pos, bool back_faces) const { return -1; }

	virtual void getNormal(const double* surface_point, double* normal) const {}

	virtual bool inside(double* point) const { return false; }

};
class Sphere : public Object {
public:

	double radius;

	Sphere(double light_radius, double pos_x, double pos_y, double pos_z, double radius, const Material* M) {
		this->light_radius = light_radius;
		this->radius = radius;
		pos[0] = pos_x;
		pos[1] = pos_y;
		pos[2] = pos_z;
		this->M = M;
		obj_type = 'S';
	}

	double getDistance(const double* view_dir, const double* view_pos, bool back_faces) const {

		// https://www.scratchapixel.com/lessons/3d-basic-rendering/minimal-ray-tracer-rendering-simple-shapes/ray-sphere-intersection.html is the solution we use. Fastest

		// Geometric solution
		double L[3] = { pos[0] - view_pos[0], pos[1] - view_pos[1], pos[2] - view_pos[2] };
		double tca = dot3D(L, view_dir);
		//if (tca < 0) return -1; // bad when you don't want to cull backfaces
		double d2 = dot3D(L, L) - tca * tca;
		//if (d2 > radius * radius) return -1; // slows down this function significantly, found this out from perf profiler
		double thc = sqrt(radius * radius - d2);
		double t0 = tca - thc;
		double t1 = tca + thc;

		if (back_faces) {
			return std::max(t0, t1);
		}
		else {
			if (t0 > t1) std::swap(t0, t1);

			if (t0 < 0) {
				t0 = t1; // If t0 is negative, let's use t1 instead.
				if (t0 < 0)
					return -1; // Both t0 and t1 are negative.
			}

			return t0;
		}

	}

	void getNormal(const double* surface_point, double* normal) const {
		for (int dim = 0; dim < 3; dim++) normal[dim] = surface_point[dim] - pos[dim];
		normalize(normal, 3);
	}

	bool inside(double* point) const {
		return power(point[0] - pos[0], 2) + power(point[1] - pos[1], 2) + power(point[2] - pos[2], 2) < radius * radius;
	}

};
class Box : public Object {
public:

	double min[3];
	double max[3];

	// size_x is actually half the total width of the cube, since size_x is added AND subtracted from center to get bounding planes
	Box(double light_radius, double pos_x, double pos_y, double pos_z, const Material* M, double size_x, double size_y, double size_z) {
		this->light_radius = light_radius;
		pos[0] = pos_x;
		pos[1] = pos_y;
		pos[2] = pos_z;
		this->M = M;
		min[0] = pos[0] - size_x;
		min[1] = pos[1] - size_y;
		min[2] = pos[2] - size_z;
		max[0] = pos[0] + size_x;
		max[1] = pos[1] + size_y;
		max[2] = pos[2] + size_z;
		obj_type = 'B';
	}

	double getDistance(const double* view_dir, const double* view_pos, bool back_faces) const {

		// https://www.scratchapixel.com/lessons/3d-basic-rendering/minimal-ray-tracer-rendering-simple-shapes/ray-box-intersection.html

		// 't' in a variable represents the idea of intersection (tzmin is minimum intersection distance from view_pos to one of the z bounding planes)

		// tmin is really txmin at the start, it just gets repurposed later, so we use its repurposed name from the start
		float tmin = (min[0] - view_pos[0]) / view_dir[0];
		float tmax = (max[0] - view_pos[0]) / view_dir[0];

		if (tmin > tmax) std::swap(tmin, tmax);

		float tymin = (min[1] - view_pos[1]) / view_dir[1];
		float tymax = (max[1] - view_pos[1]) / view_dir[1];

		if (tymin > tymax) std::swap(tymin, tymax);

		if ((tmin > tymax) || (tymin > tmax)) return -1;

		if (tymin > tmin) tmin = tymin;
		if (tymax < tmax) tmax = tymax;

		float tzmin = (min[2] - view_pos[2]) / view_dir[2];
		float tzmax = (max[2] - view_pos[2]) / view_dir[2];

		if (tzmin > tzmax) std::swap(tzmin, tzmax);

		if ((tmin > tzmax) || (tzmin > tmax)) return -1;

		if (tzmin > tmin) tmin = tzmin;
		if (tzmax < tmax) tmax = tzmax;

		return std::min(tmin, tmax);
	}

	void getNormal(const double* surface_point, double* normal) const {

		// distance from min and max in each dimension
		double distances[6];
		for (int dim = 0; dim < 3; dim++) {
			distances[dim * 2 + 0] = surface_point[dim] - min[dim];
			distances[dim * 2 + 1] = max[dim] - surface_point[dim];
		}

		// finding the side with the closest distance (in its dimension, to the surface point)
		double closest_distance = NAN;
		int closest_side = 0;
		for (int side = 0; side < 6; side++) {
			if (!(distances[side] > closest_distance)) {
				closest_distance = distances[side];
				closest_side = side;
			}
		}

		// closest side affects that dimension of normal, all other dimensions of normal are zero
		for (int dim = 0; dim < 3; dim++) {
			if (closest_side == dim * 2 + 0) normal[dim] = -1;
			else if (closest_side == dim * 2 + 1) normal[dim] = 1;
			else normal[dim] = 0;
		}

		normalize(normal, 3);

	}

	bool inside(double* position) const {
		return position[0] > min[0] && position[1] > min[1] && position[2] > min[2] &&
			position[0] < max[0] && position[1] < max[1] && position[2] < max[2];
	}

};
class Group : public Object {
public:
	double radius = 0; // must be same location as Sphere
	Object** objects = nullptr; // first object is always the one whose volume remains in even of overlap
	int num_objects = 0;
	bool Union = false;
	bool Intersection = false;
	bool Difference = false; // I, U, D for intersection, union, difference

	Group() {}

	Group(Object** objects, int num_objects, int combine_method, double group_radius, double pos_x, double pos_y, double pos_z, const Material* M) : objects(objects), num_objects(num_objects) { // 0 for Union, 1 for intersection, 2 for Difference
		Union = (combine_method == 0);
		Intersection = (combine_method == 1);
		Difference = (combine_method == 2);
		radius = group_radius;
		pos[0] = pos_x;
		pos[1] = pos_y;
		pos[2] = pos_z;
		this->M = (Material*)M;
		for (int child_idx = 0; child_idx < num_objects; child_idx++) objects[child_idx]->parent = this;
		obj_type = 'G';
	}

	// distance to this group's bounding sphere
	double getDistance(const double* view_dir, const double* view_pos, bool back_faces) const {

		return ((Sphere*)this)->getDistance(view_dir, view_dir, true);

	}

	// given view dir and pos, find m2, intersection_dist, normal. m2 is returned, the rest are passed by ref
	const Material* getIntersect(const Material* m1, double* intersection_dist, double* normal, double* view_dir, double* view_pos, double* data) const {

		// get info for each group and shape object
		const Material* closest_object_material = nullptr;
		for (int child_idx = 0; child_idx < num_objects; child_idx++) {
			double dist = objects[child_idx]->getDistance(view_dir, view_pos, m1 == objects[child_idx]->M);

			if (dist > 0 && objects[child_idx]->obj_type == 'G') {
				//return ((Group*)objects[child_idx])->getIntersect(m1, intersection_dist, normal, view_dir, view_pos, data);
			}
			else {
				if (((dist < (*intersection_dist) && dist >= 0) || ((*intersection_dist) < 0 && dist >= 0))) {

					double intersection_point[3];
					for (int dim = 0; dim < 3; dim++) {
						intersection_point[dim] = view_pos[dim] + view_dir[dim] * dist;
					}

					double temp_normal[3];
					objects[child_idx]->getNormal(intersection_point, temp_normal);

					// if intersect material not same material as what ray is inside of, intersect normal must be opposite of view_dir
					// if intersect material IS material ray is inside of, normal must be same as view dir (ie the ray must be exiting the material)
					if (objects[child_idx]->M != m1 && dot3D(temp_normal, view_dir) > 0) {}
					else if (objects[child_idx]->M == m1 && dot3D(temp_normal, view_dir) < 0) {}
					else {
						double intersection_point[3];
						for (int dim = 0; dim < 3; dim++) {
							intersection_point[dim] = view_pos[dim] + view_dir[dim] * dist;
						}

						(*intersection_dist) = dist;
						objects[child_idx]->getNormal(intersection_point, normal);
						closest_object_material = objects[child_idx]->M;
					}

				}
			}

		}

		return closest_object_material;

	}

	bool inside(double* point) const {
		return power(point[0] - pos[0], 2) + power(point[1] - pos[1], 2) + power(point[2] - pos[2], 2) < radius * radius;
	}

};

Object* adam_objects[3] = {
	new Box(.1, .3, 0, 3.5, &light, .1, .1, .1),
	new Box(.1, .3, .3, 3.5, &light2, .1, .1, .1),
	new Sphere(.1, .3, 0, 1, .2, &wall)
};

Group adam(
	(Object**)adam_objects, 3,
	0, // combine method
	100, // bounding sphere radius
	0, 0, 0, // bounding sphere position
	&air // material index
);

const Material* materialAt(Object* cur_obj, double* point, const Material* blacklisted) {

	Group* cur_obj_as_group = (Group*)cur_obj;

	while (cur_obj->obj_type == 'G') {

		for (int child_idx = 0; child_idx < cur_obj_as_group->num_objects; child_idx++) {

			if (cur_obj_as_group->objects[child_idx]->M != blacklisted && cur_obj_as_group->objects[child_idx]->inside(point)) {
				cur_obj = cur_obj_as_group->objects[child_idx];
				cur_obj_as_group = (Group*)cur_obj;
			}

		}

		break;

	}

	return cur_obj->M;

}

void findColor(const Material* m1, double* view_dir, double* view_pos, double factor, double* image_data, uint64_t* rand_seed) {
	
	double local_view_dir[3] = { view_dir[0], view_dir[1], view_dir[2] };
	double local_view_pos[3] = { view_pos[0], view_pos[1], view_pos[2] };

	for (int ray_depth = 0; ray_depth <= 2; ray_depth++) {

		// initialize values needed for light calculation: m1 and m2, 
		const Material* m2 = nullptr;
		double intersection_dist = -1;
		double normal[3] = { 0, 0, 0 }; // normal at intersection

		m2 = adam.getIntersect(m1, &intersection_dist, normal, local_view_dir, local_view_pos, image_data);

		// get intersection point of view ray with object so we can do other calculations using it
		double intersection_point[3];
		for (int channel = 0; channel < 3; channel++) {
			intersection_point[channel] = local_view_pos[channel] + local_view_dir[channel] * intersection_dist;
		}

		// if ray is transmitting through material and hitting a backface, reverse object surface normal
		if (m1 == m2 && dot3D(local_view_dir, normal) > 0) {
			for (int dim = 0; dim < 3; dim++) normal[dim] *= -1;
		}

		// if ray inside of non-air material AND hitting its own boundary (ray exiting its own boundary)
		if (m1 != adam.M && m1 == m2) {
			m2 = materialAt((Object*) & adam, intersection_point, m1);
			if (m2 == nullptr) m2 = adam.M;
		}

		// if intersection point not found, return early so that no additional color is added (ie. background color is black)
		if (intersection_dist <= 0) return;

		// add emission light
		for (int channel = 0; channel < 3; channel++) image_data[channel] += m2->emissivity * m2->color[channel] * factor;

		// add reflected light

		// first, find reflection dir (using formula from https://www.scratchapixel.com/lessons/3d-basic-rendering/introduction-to-shading/reflection-refraction-fresnel.html)
		double view_reflect_dir[3];
		double temp = dot3D(local_view_dir, normal);
		for (int dim = 0; dim < 3; dim++) {
			view_reflect_dir[dim] = local_view_dir[dim] - 2 * temp * normal[dim];
		}
		normalize(view_reflect_dir, 3);

		// begin fresnel equations by finding theta_1 and theta_2 (snell's law)
		double intersection_eye_vector[3];
		for (int dim = 0; dim < 3; dim++) intersection_eye_vector[dim] = local_view_pos[dim] - intersection_point[dim];
		double theta_1 = getAngleVectors(normal, intersection_eye_vector);
		double theta_2 = asin(m1->refr_index * sin(theta_1) / m2->refr_index);

		double n1costheta1 = m1->refr_index * cos(theta_1);
		double n1costheta2 = m1->refr_index * cos(theta_2);
		double n2costheta1 = m2->refr_index * cos(theta_1);
		double n2costheta2 = m2->refr_index * cos(theta_2);

		// calculate reflection factor using fresnel equations
		double rs = (n1costheta1 - n2costheta2) / (n1costheta1 + n2costheta2);
		double rp = (n1costheta2 - n2costheta1) / (n1costheta2 + n2costheta1);
		double reflection_factor = (.5 * rs * rs + .5 * rp * rp);


		// next, get a random direction
		double rand_dir[3];
		randDirection(rand_dir, rand_seed);
		if (dot3D(rand_dir, normal) < 0) {
			for (int dim = 0; dim < 3; dim++) rand_dir[dim] *= -1;
		}

		// now, onto transmission
		double c1 = cos(theta_1);
		double c2 = sqrt(1 - power(m1->refr_index / m2->refr_index, 2) * power(sin(theta_1), 2));

		double view_transmit_dir[3];
		for (int dim = 0; dim < 3; dim++) {
			view_transmit_dir[dim] = m1->refr_index * (local_view_dir[dim] + c1 * normal[dim]) - normal[dim] * c2;
		}
		normalize(view_transmit_dir, 3);

		// apply roughness to transmission and reflection vectors
		if (m2 == &air) {
			for (int dim = 0; dim < 3; dim++) view_reflect_dir[dim] = m1->roughness * rand_dir[dim] + (1 - m1->roughness) * view_reflect_dir[dim];
			for (int dim = 0; dim < 3; dim++) view_transmit_dir[dim] = m1->roughness * -rand_dir[dim] + (1 - m1->roughness) * view_transmit_dir[dim];
		}
		else {
			for (int dim = 0; dim < 3; dim++) view_reflect_dir[dim] = m2->roughness * rand_dir[dim] + (1 - m2->roughness) * view_reflect_dir[dim];
			for (int dim = 0; dim < 3; dim++) view_transmit_dir[dim] = m2->roughness * -rand_dir[dim] + (1 - m2->roughness) * view_transmit_dir[dim];
		}
		normalize(view_reflect_dir, 3);
		normalize(view_transmit_dir, 3);

		// DEBUG
		if (ray_depth == 1 && false) {
			image_data[0] = (view_transmit_dir[0] + 1) / 2 * 255;
			image_data[1] = (view_transmit_dir[1] + 1) / 2 * 255;
			image_data[2] = (view_transmit_dir[2] + 1) / 2 * 255;
			return;
		}

		// apply transmission OR reflection

		// remember to specify all variables used as input to findColor() function. define m1 and factor if they change! (factor should change!!!)
		if (randDouble(rand_seed) < reflection_factor) {
			for (int dim = 0; dim < 3; dim++) {
				local_view_dir[dim] = view_reflect_dir[dim];
				local_view_pos[dim] = intersection_point[dim];
			}
		}
		else {
			m1 = m2;
			for (int dim = 0; dim < 3; dim++) {
				local_view_dir[dim] = view_transmit_dir[dim];
				local_view_pos[dim] = intersection_point[dim];
			}
		}

	}

}

//#define GPU_ENABLE

#ifdef GPU_ENABLE
void func(int WIDTH, int HEIGHT, unsigned char* image_data, double* image_data_double, int frames_still, double* eye_rotation, double* eye_pos, uint64_t* rand_seeds) {
    sycl::queue q;
    sycl::buffer<unsigned char, 1> imageBuffer((unsigned char*)image_data, sycl::range<1>(WIDTH * HEIGHT * 3));
	sycl::buffer<double, 1> imageBuffer_double((double*)image_data_double, sycl::range<1>(WIDTH * HEIGHT * 3));

    q.submit([&](sycl::handler& h) {
		auto image = imageBuffer.get_access<sycl::access::mode::read_write>(h);
		auto image_double = imageBuffer_double.get_access<sycl::access::mode::read_write>(h);
        h.parallel_for(sycl::range<1>(WIDTH * HEIGHT), [=](sycl::id<1> idx) {
            int w = idx % WIDTH;
            int h = idx / WIDTH;

			// calculate this fragment's direction
			double frag_dir[3] = { 2.0 * w / WIDTH - 1, 2.0 * h / HEIGHT - 1, 1 };
			rotateX(frag_dir, eye_rotation[0]);
			rotateY(frag_dir, eye_rotation[1]);
			normalize(frag_dir, 3);

			// call raytracing function to get this fragment's color
			double frag_color[3] = { 0, 0, 0 };

			findColor((adam.M), frag_dir, eye_pos, 1.0, frag_color, &(rand_seeds[idx]));
			for (int channel = 0; channel < 3; channel++) {
				image_double[(h * WIDTH + w) * 3 + channel] *= frames_still / (frames_still + 1.0);
				image_double[(h * WIDTH + w) * 3 + channel] += frag_color[channel] / (frames_still + 1.0);
				image[(h * WIDTH + w) * 3 + channel] = std::min((int)image_double[(h * WIDTH + w) * 3 + channel], 255);
			}

            });
        }).wait();
}
#else
void func(int WIDTH, int HEIGHT, unsigned char* image_data, double* image_data_double, int frames_still, double* eye_rotation, double* eye_pos, uint64_t* rand_seeds) {
	
	// for each fragment (each pixel on screen)...
	for (int pixel_index = 0; pixel_index < WIDTH * HEIGHT; pixel_index++) {
		int w = pixel_index % WIDTH;
		int h = pixel_index / WIDTH;
		// calculate this fragment's direction
		double frag_dir[3] = { 2.0 * w / WIDTH - 1, 2.0 * h / HEIGHT - 1, 1 };
		rotateX(frag_dir, eye_rotation[0]);
		rotateY(frag_dir, eye_rotation[1]);
		normalize(frag_dir, 3);

		// call raytracing function to get this fragment's color
		double frag_color[3] = { 0, 0, 0 };
		findColor((adam.M), frag_dir, eye_pos, 1.0, frag_color, &(rand_seeds[pixel_index]));
		for (int channel = 0; channel < 3; channel++) {
			image_data_double[(h * WIDTH + w) * 3 + channel] *= frames_still / (frames_still + 1.0);
			image_data_double[(h * WIDTH + w) * 3 + channel] += frag_color[channel] / (frames_still + 1.0);
			image_data[(h * WIDTH + w) * 3 + channel] = std::min((int)image_data_double[(h * WIDTH + w) * 3 + channel], 255);
		}
	}

}
#endif

