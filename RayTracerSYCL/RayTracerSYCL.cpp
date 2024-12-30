#include <fstream>
#include <sycl/sycl.hpp>
#include "PRNGs.h"
#include "RayTracerSYCL.h"

constexpr float PI = 3.14159265358979323846f; // can't use M_PI because M_PI is of type double. This is also just easier

// GOALS
// light through filament (see previous commit)
// scaling (harder than initially thought. Why?)

// Create a queue using a GPU selector
sycl::queue Q{ sycl::gpu_selector_v };

// faster pow() function for when using positive integer powers
float power(float x, int p) {
	float product = 1.0f;
	for (int i = 0; i < p; i++) product *= x;
	return product;
}
// simple dot product function
float dot3D(const float* a, const float* b) {
	return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
// simple normalization function
void normalize(float* v, int dim) {
	float length = 0.0f;
	for (int i = 0; i < dim; i++) {
		length += power(v[i], 2);
	}
	length = sqrt(length);
	for (int i = 0; i < dim; i++) {
		v[i] /= length;
	}
}
// rotates 3d vector around x-axis
void rotateX(float* v, float angle) { // angle in radians
	float temp[3] = { v[0], cos(angle) * v[1] - sin(angle) * v[2], sin(angle) * v[1] + cos(angle) * v[2] };
	v[0] = temp[0];
	v[1] = temp[1];
	v[2] = temp[2];
}
// rotates 3d vector around y-axis
void rotateY(float* v, float angle) { // angle in radians
	float temp[3] = {
		cos(angle) * v[0] + sin(angle) * v[2],
		v[1],
		-sin(angle) * v[0] + cos(angle) * v[2]
	};
	v[0] = temp[0];
	v[1] = temp[1];
	v[2] = temp[2];
}
float vectorLength3D(const float* V) {
	return sqrt(V[0] * V[0] + V[1] * V[1] + V[2] * V[2]);
}
float distance3D(const float* A, const float* B) {
	float C[3] = { A[0] - B[0], A[1] - B[1], A[2] - B[2] };
	return vectorLength3D(C);
}
float getAngle(const float* A, const float* B, const float* C) { // B is crux of angle
	float BA[3]; // vector from B to A
	float BC[3]; // vector from B to C
	for (int channel = 0; channel < 3; channel++) {
		BA[channel] = A[channel] - B[channel];
		BC[channel] = C[channel] - B[channel];
	}
	return acos(dot3D(BA, BC) / (vectorLength3D(BA) * vectorLength3D(BC)));
}
float getAngleVectors(const float* BA, const float* BC) { // B is crux of angle
	return acos(dot3D(BA, BC) / (vectorLength3D(BA) * vectorLength3D(BC)));
}

void randDirection(float* dir, uint32_t* rand_seed) {

	// from here: https://stackoverflow.com/questions/5408276/sampling-uniformly-distributed-random-points-inside-a-spherical-volume

	float phi = randFloat(rand_seed) * PI * 2.0f;
	float costheta = randFloat(rand_seed) * 2.0f - 1.0f;

	float theta = acos(costheta);
	dir[0] = sin(theta) * cos(phi);
	dir[1] = sin(theta) * sin(phi);
	dir[2] = cos(theta);

}

// GLOBAL VARIABLES
std::ofstream debug_file("debug_output.txt", std::ofstream::out);

// create materials
class Material {
public:
	float color[3];				// Only used when sphere is emissive
	float opt_den[3];			// optical density; Beer-Lambert law (range 0-1)
	float emissivity;
	float refr_index;
	float roughness;
	float general_optical_density;
	float diffusion_distance;	// how far a diffused ray travels through this material before exiting

	// Make the constructor constexpr for compile-time evaluation
	constexpr Material(float color_r, float color_g, float color_b, float emissivity,
		float opt_den_r, float opt_den_g, float opt_den_b,
		float roughness, float refr_index, float general_optical_density, float diffusion_distance)
		: color{ color_r, color_g, color_b },
		opt_den{ opt_den_r, opt_den_g, opt_den_b },
		emissivity(emissivity),
		refr_index(refr_index),
		roughness(roughness),
		general_optical_density(general_optical_density),
		diffusion_distance(diffusion_distance) {}
};

#define GPU_ENABLE

#ifdef GPU_ENABLE

class Object {
public:
	float pos[3];
	float scale[3];
	int M;
	char obj_type;

	Object(float pos_x, float pos_y, float pos_z, float scale_x, float scale_y, float scale_z, int mat_index, char obj_type) {
		pos[0] = pos_x;
		pos[1] = pos_y;
		pos[2] = pos_z;

		scale[0] = scale_x;
		scale[1] = scale_y;
		scale[2] = scale_z;

		this->M = mat_index;
		this->obj_type = obj_type;
	}

	void pointRealToIdeal(float* real_point, float* ideal_point) {
		for (int dim = 0; dim < 3; dim++) {
			ideal_point[dim] = real_point[dim] - pos[dim];
		}
	}

	float getDistance(float* view_dir, float* view_pos, bool back_faces) {

		float local_view_pos[3];
		pointRealToIdeal(view_pos, local_view_pos);

		if (obj_type == 'S') {
			// https://www.scratchapixel.com/lessons/3d-basic-rendering/minimal-ray-tracer-rendering-simple-shapes/ray-sphere-intersection.html is the solution we use. Fastest

			// Geometric solution
			float L[3] = { 0 - local_view_pos[0], 0 - local_view_pos[1], 0 - local_view_pos[2] }; // position assumed zero in "ideal" space
			float tca = dot3D(L, view_dir);
			//if (tca < 0) return -1; // bad when you don't want to cull backfaces
			float d2 = dot3D(L, L) - tca * tca;
			//if (d2 > radius * radius) return -1; // slows down this function significantly, found this out from perf profiler
			float thc = sqrt(1.0f * 1.0f - d2); // radius assumed 1 in "ideal" space
			float t0 = tca - thc;
			float t1 = tca + thc;

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
		else if (obj_type == 'B') {

			// https://www.scratchapixel.com/lessons/3d-basic-rendering/minimal-ray-tracer-rendering-simple-shapes/ray-box-intersection.html

			// 't' in a variable represents the idea of intersection (tzmin is minimum intersection distance from view_pos to one of the z bounding planes)

			// tmin is really txmin at the start, it just gets repurposed later, so we use its repurposed name from the start
			float tmin = (-1 - local_view_pos[0]) / view_dir[0];
			float tmax = (1 - local_view_pos[0]) / view_dir[0];

			if (tmin > tmax) std::swap(tmin, tmax);

			float tymin = (-1 - local_view_pos[1]) / view_dir[1];
			float tymax = (1 - local_view_pos[1]) / view_dir[1];

			if (tymin > tymax) std::swap(tymin, tymax);

			if ((tmin > tymax) || (tymin > tmax)) return -1;

			if (tymin > tmin) tmin = tymin;
			if (tymax < tmax) tmax = tymax;

			float tzmin = (-1 - local_view_pos[2]) / view_dir[2];
			float tzmax = (1 - local_view_pos[2]) / view_dir[2];

			if (tzmin > tzmax) std::swap(tzmin, tzmax);

			if ((tmin > tzmax) || (tzmin > tmax)) return -1;

			if (tzmin > tmin) tmin = tzmin;
			if (tzmax < tmax) tmax = tzmax;

			return std::min(tmin, tmax);

		}

		// we should never reach this point
		return -1;

	}

	bool inside(float* point) {

		float local_point[3];
		pointRealToIdeal(point, local_point);

		if (obj_type == 'S') {
			return power(local_point[0], 2) + power(local_point[1], 2) + power(local_point[2], 2) < 1.0f * 1.0f; // radius assumed 1
		}
		else if (obj_type == 'B') {
			return local_point[0] > -1 && local_point[1] > -1 && local_point[2] > -1 &&
				local_point[0] < 1 && local_point[1] < 1 && local_point[2] < 1;
		}

		// we should never reach this point
		return false;

	}

	void getNormal(float* surface_point, float* normal) {

		float local_surface_point[3];
		pointRealToIdeal(surface_point, local_surface_point);

		if (obj_type == 'S') {
			for (int dim = 0; dim < 3; dim++) normal[dim] = local_surface_point[dim];
		}
		else if (obj_type == 'B') {

			// distance from min and max in each dimension
			float distances[6];
			for (int dim = 0; dim < 3; dim++) {
				distances[dim * 2 + 0] = local_surface_point[dim] - (-1);
				distances[dim * 2 + 1] = 1 - local_surface_point[dim];
			}

			// finding the side with the closest distance (in its dimension, to the surface point)
			float closest_distance = NAN;
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

		}

		normalize(normal, 3);

	}

};

// given view dir and pos, find m2, intersection_dist, normal. m2 is returned, the rest are passed by ref
void getIntersect(Object* object_list, int num_objects, int m1, int* m2, float* intersection_point, float* intersection_dist, float* normal, float* view_dir, float* view_pos, float* data) {
	//if (debug) return;
	// get info for each group and shape object
	(*m2) = -1;
	for (int child_idx = 0; child_idx < num_objects; child_idx++) {
		float dist = object_list[child_idx].getDistance(view_dir, view_pos, m1 == object_list[child_idx].M);

		if (((dist < (*intersection_dist) && dist >= 0) || ((*intersection_dist) < 0 && dist >= 0))) {

			for (int dim = 0; dim < 3; dim++) {
				intersection_point[dim] = view_pos[dim] + view_dir[dim] * dist;
			}

			float temp_normal[3];
			object_list[child_idx].getNormal(intersection_point, temp_normal);

			// if intersect material not same material as what ray is inside of, intersect normal must be opposite of view_dir
			// if intersect material IS material ray is inside of, normal must be same as view dir (ie the ray must be exiting the material)
			if (object_list[child_idx].M != m1 && dot3D(temp_normal, view_dir) > 0) {}
			else if (object_list[child_idx].M == m1 && dot3D(temp_normal, view_dir) < 0) {}
			else {
				(*intersection_dist) = dist;
				object_list[child_idx].getNormal(intersection_point, normal);
				(*m2) = object_list[child_idx].M;
			}

		}

	}

}

int materialAt(Object* object_list, int num_objects, float* point, int blacklisted_mat, int air_mat) {

	for (int sphere_index = 0; sphere_index < num_objects; sphere_index++) {
	
		if (object_list[sphere_index].inside(point) && object_list[sphere_index].M != blacklisted_mat) {
			return object_list[sphere_index].M;
		}
	
	}

	return air_mat;

}

void findColor(Object* object_list, int num_objects, Material* mats, float* view_dir, float* view_pos, float* factors, float* image_data, uint32_t* rand_seed, int air_mat) {

	float local_view_dir[3] = { view_dir[0], view_dir[1], view_dir[2] };
	float local_view_pos[3] = { view_pos[0], view_pos[1], view_pos[2] };
	int m1 = materialAt(object_list, num_objects, view_pos, -1, air_mat);

	for (int ray_depth = 0; ray_depth <= 2; ray_depth++) {

		// initialize values needed for light calculation: m1 and m2, 
		int m2 = -1; // material being hit by ray
		float normal[3] = { 0, 0, 0 }; // normal at intersection
		float intersection_point[3]; // point in real space, not "ideal" space
		float intersection_dist = -1; // real distance, not distance in "ideal" space

		

		// get intersection material, normal, intersection point
		getIntersect(object_list, num_objects, m1, &m2, intersection_point, &intersection_dist, normal, local_view_dir, local_view_pos, image_data);

		// this must be updated after getIntersect, trust me
		for (int dim = 0; dim < 3; dim++) {
			intersection_point[dim] = local_view_pos[dim] + local_view_dir[dim] * intersection_dist;
		}

		// if intersection point not found, return early so that no additional color is added (ie. background color is black)
		if (intersection_dist <= 0) {
			break;
		}

		// use m1 and intersection dist to apply Beer-Lambert law
		for (int channel = 0; channel < 3; channel++) {
			factors[channel] *= exp(-mats[m1].opt_den[channel] * intersection_dist * mats[m1].general_optical_density);
		}

		// if ray is transmitting through material and hitting a backface, reverse object surface normal
		if (m1 == m2 && dot3D(local_view_dir, normal) > 0) {
			for (int dim = 0; dim < 3; dim++) normal[dim] *= -1;
		}

		// if ray inside of non-air material AND hitting its own boundary (ray exiting its own boundary)
		if (m1 != air_mat && m1 == m2) {
			m2 = materialAt(object_list, num_objects, intersection_point, m1, air_mat);
		}

		// add emission light
		for (int channel = 0; channel < 3; channel++) image_data[channel] += mats[m2].emissivity * mats[m2].color[channel] * factors[channel];

		// add reflected light

		// first, find reflection dir (using formula from https://www.scratchapixel.com/lessons/3d-basic-rendering/introduction-to-shading/reflection-refraction-fresnel.html)
		float view_reflect_dir[3];
		float temp = dot3D(local_view_dir, normal);
		for (int dim = 0; dim < 3; dim++) {
			view_reflect_dir[dim] = local_view_dir[dim] - 2 * temp * normal[dim];
		}
		normalize(view_reflect_dir, 3);

		// begin fresnel equations by finding theta_1 and theta_2 (snell's law)
		float intersection_eye_vector[3];
		for (int dim = 0; dim < 3; dim++) intersection_eye_vector[dim] = local_view_pos[dim] - intersection_point[dim];
		normalize(intersection_eye_vector, 3);
		float theta_1 = getAngleVectors(normal, intersection_eye_vector);
		float theta_2 = asin(mats[m1].refr_index * sin(theta_1) / mats[m2].refr_index);

		float n1costheta1 = mats[m1].refr_index * cos(theta_1);
		float n1costheta2 = mats[m1].refr_index * cos(theta_2);
		float n2costheta1 = mats[m2].refr_index * cos(theta_1);
		float n2costheta2 = mats[m2].refr_index * cos(theta_2);

		// calculate reflection factor using fresnel equations
		float rs = (n1costheta1 - n2costheta2) / (n1costheta1 + n2costheta2);
		float rp = (n1costheta2 - n2costheta1) / (n1costheta2 + n2costheta1);
		float reflection_factor = (.5f * rs * rs + .5f * rp * rp);


		// next, get a random direction
		float rand_dir[3];
		randDirection(rand_dir, rand_seed);
		if (dot3D(rand_dir, normal) < 0) {
			for (int dim = 0; dim < 3; dim++) rand_dir[dim] *= -1;
		}


		// now, onto transmission
		float c1 = cos(theta_1);
		float c2 = sqrt(1 - power(mats[m1].refr_index / mats[m2].refr_index, 2) * power(sin(theta_1), 2));

		float view_transmit_dir[3];
		for (int dim = 0; dim < 3; dim++) {
			view_transmit_dir[dim] = (mats[m1].refr_index / mats[m2].refr_index) * (local_view_dir[dim] + c1 * normal[dim]) - normal[dim] * c2;
		}
		normalize(view_transmit_dir, 3);

		// apply roughness to transmission and reflection vectors
		if (m2 == air_mat) {
			for (int dim = 0; dim < 3; dim++) view_reflect_dir[dim] = mats[m1].roughness * rand_dir[dim] + (1 - mats[m1].roughness) * view_reflect_dir[dim];
			for (int dim = 0; dim < 3; dim++) view_transmit_dir[dim] = mats[m1].roughness * -rand_dir[dim] + (1 - mats[m1].roughness) * view_transmit_dir[dim];
		}
		else {
			for (int dim = 0; dim < 3; dim++) view_reflect_dir[dim] = mats[m2].roughness * rand_dir[dim] + (1 - mats[m2].roughness) * view_reflect_dir[dim];
			for (int dim = 0; dim < 3; dim++) view_transmit_dir[dim] = mats[m2].roughness * -rand_dir[dim] + (1 - mats[m2].roughness) * view_transmit_dir[dim];
		}
		normalize(view_reflect_dir, 3);
		normalize(view_transmit_dir, 3);

		// apply transmission OR reflection

		// remember to specify all variables used as input to findColor() function. define m1 and factor if they change! (factor should change!!!)
		if (randFloat(rand_seed) <= reflection_factor) {

			// Beer-Lambert law, applied to diffuse reflection, using m2 optical density and diffusion distance (and roughness)
			for (int channel = 0; channel < 3; channel++) {
				factors[channel] *= exp(-mats[m2].opt_den[channel] * mats[m2].diffusion_distance * mats[m2].general_optical_density * mats[m2].roughness);
			}

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

	// add background emission
	if (m1 == air_mat){
		for (int channel = 0; channel < 3; channel++) image_data[channel] += 100 * factors[channel];
	}


}

#else

constexpr Material light(255, 255, 255, 1.0f, // light
	255, 255, 255, 1.0f,
	1.5f);
constexpr Material wall(255, 255, 255, 0.0f, // wall
	255, 0, 255, 0.0f,
	1.33f);
constexpr Material air(255, 255, 255, 0.0f, // air
	255, 0, 255, 0.5f,
	1.0f);
constexpr Material light2(0, 255, 255, 1.0f, // light2
	255, 255, 255, 1.0f,
	1.5f);

class Object {
public:

	float pos[3];
	const Material* M;
	float light_radius; // when an object is emissive, for ambient lighting it is treated as a sphere of this radius
	void* parent = nullptr; // cast to group pointer
	int precedence;
	char obj_type; // group, sphere, box, etc.
	float radius;
	float min[3];
	float max[3];
	Object* objects = nullptr;
	int num_objects = 0;
	bool Union = false;
	bool Intersection = false;
	bool Difference = false; // I, U, D for intersection, union, difference

	Object() {}

	Object(float light_radius, float pos_x, float pos_y, float pos_z, float radius, const Material* M) {
		this->light_radius = light_radius;
		this->radius = radius;
		pos[0] = pos_x;
		pos[1] = pos_y;
		pos[2] = pos_z;
		this->M = M;
		obj_type = 'S';
	}

	Object(float light_radius, float pos_x, float pos_y, float pos_z, const Material* M, float size_x, float size_y, float size_z) {
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

	Object(Object* objects, int num_objects, int combine_method, float group_radius, float pos_x, float pos_y, float pos_z, const Material* M) : objects(objects), num_objects(num_objects) { // 0 for Union, 1 for intersection, 2 for Difference
		Union = (combine_method == 0);
		Intersection = (combine_method == 1);
		Difference = (combine_method == 2);
		radius = group_radius;
		pos[0] = pos_x;
		pos[1] = pos_y;
		pos[2] = pos_z;
		this->M = (Material*)M;
		for (int child_idx = 0; child_idx < num_objects; child_idx++) objects[child_idx].parent = this;
		obj_type = 'G';
	}

	float getDistance(const float* view_dir, const float* view_pos, bool back_faces) const {

		if (obj_type == 'S' || obj_type == 'G') {

			// https://www.scratchapixel.com/lessons/3d-basic-rendering/minimal-ray-tracer-rendering-simple-shapes/ray-sphere-intersection.html is the solution we use. Fastest

			// Geometric solution
			float L[3] = { pos[0] - view_pos[0], pos[1] - view_pos[1], pos[2] - view_pos[2] };
			float tca = dot3D(L, view_dir);
			//if (tca < 0) return -1; // bad when you don't want to cull backfaces
			float d2 = dot3D(L, L) - tca * tca;
			//if (d2 > radius * radius) return -1; // slows down this function significantly, found this out from perf profiler
			float thc = sqrt(radius * radius - d2);
			float t0 = tca - thc;
			float t1 = tca + thc;

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
		else if (obj_type == 'B') {

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


	}

	void getNormal(const float* surface_point, float* normal) const {

		if (obj_type == 'S') {

			for (int dim = 0; dim < 3; dim++) normal[dim] = surface_point[dim] - pos[dim];

		}
		else if (obj_type == 'B') {

			// distance from min and max in each dimension
			float distances[6];
			for (int dim = 0; dim < 3; dim++) {
				distances[dim * 2 + 0] = surface_point[dim] - min[dim];
				distances[dim * 2 + 1] = max[dim] - surface_point[dim];
			}

			// finding the side with the closest distance (in its dimension, to the surface point)
			float closest_distance = NAN;
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

		}

		normalize(normal, 3);

	}

	bool inside(float* point) const {

		if (obj_type == 'S' || obj_type == 'G') {
			return power(point[0] - pos[0], 2) + power(point[1] - pos[1], 2) + power(point[2] - pos[2], 2) < radius * radius;
		}
		else if (obj_type == 'B') {
			return point[0] > min[0] && point[1] > min[1] && point[2] > min[2] &&
				point[0] < max[0] && point[1] < max[1] && point[2] < max[2];
		}

	}

	// given view dir and pos, find m2, intersection_dist, normal. m2 is returned, the rest are passed by ref
	const Material* getIntersect(const Material* m1, float* intersection_dist, float* normal, float* view_dir, float* view_pos, float* data) const {

		// get info for each group and shape object
		const Material* closest_object_material = nullptr;
		for (int child_idx = 0; child_idx < num_objects; child_idx++) {
			float dist = objects[child_idx].getDistance(view_dir, view_pos, m1 == objects[child_idx].M);

			if (dist > 0 && objects[child_idx].obj_type == 'G') {
				//return ((Object*)objects[child_idx])->getIntersect(m1, intersection_dist, normal, view_dir, view_pos, data);
			}
			else {
				if (((dist < (*intersection_dist) && dist >= 0) || ((*intersection_dist) < 0 && dist >= 0))) {

					float intersection_point[3];
					for (int dim = 0; dim < 3; dim++) {
						intersection_point[dim] = view_pos[dim] + view_dir[dim] * dist;
					}

					float temp_normal[3];
					objects[child_idx].getNormal(intersection_point, temp_normal);

					// if intersect material not same material as what ray is inside of, intersect normal must be opposite of view_dir
					// if intersect material IS material ray is inside of, normal must be same as view dir (ie the ray must be exiting the material)
					if (objects[child_idx].M != m1 && dot3D(temp_normal, view_dir) > 0) {}
					else if (objects[child_idx].M == m1 && dot3D(temp_normal, view_dir) < 0) {}
					else {
						(*intersection_dist) = dist;
						objects[child_idx].getNormal(intersection_point, normal);
						closest_object_material = objects[child_idx].M;
					}

				}
			}

		}

		return closest_object_material;

	}


};

// Basically say "Object isn't implicitly defined as device-copyable, but trust me, it is" (it really isn't at the moment)
template<>
struct sycl::is_device_copyable<Object> : std::true_type {};

const Material* materialAt(Object* cur_obj, float* point, const Material* blacklisted) {

	Object* cur_obj_as_group = (Object*)cur_obj;

	while (cur_obj->obj_type == 'G') {

		for (int child_idx = 0; child_idx < cur_obj_as_group->num_objects; child_idx++) {

			if (cur_obj_as_group->objects[child_idx].M != blacklisted && cur_obj_as_group->objects[child_idx].inside(point)) {
				cur_obj = &(cur_obj_as_group->objects[child_idx]);
				cur_obj_as_group = (Object*)cur_obj;
			}

		}

		break;

	}

	return cur_obj->M;

}

void findColor(Object* adam, float* view_dir, float* view_pos, float factor, float* image_data, uint32_t* rand_seed) {

	float local_view_dir[3] = { view_dir[0], view_dir[1], view_dir[2] };
	float local_view_pos[3] = { view_pos[0], view_pos[1], view_pos[2] };
	const Material* m1 = adam->M;

	for (int ray_depth = 0; ray_depth <= 2; ray_depth++) {

		// initialize values needed for light calculation: m1 and m2, 
		const Material* m2 = nullptr;
		float intersection_dist = -1;
		float normal[3] = { 0, 0, 0 }; // normal at intersection

		m2 = adam->getIntersect(m1, &intersection_dist, normal, local_view_dir, local_view_pos, image_data);

		// get intersection point of view ray with object so we can do other calculations using it
		float intersection_point[3];
		for (int channel = 0; channel < 3; channel++) {
			intersection_point[channel] = local_view_pos[channel] + local_view_dir[channel] * intersection_dist;
		}

		// if ray is transmitting through material and hitting a backface, reverse object surface normal
		if (m1 == m2 && dot3D(local_view_dir, normal) > 0) {
			for (int dim = 0; dim < 3; dim++) normal[dim] *= -1;
		}

		// if ray inside of non-air material AND hitting its own boundary (ray exiting its own boundary)
		if (m1 != adam->M && m1 == m2) {
			m2 = materialAt((Object*)adam, intersection_point, m1);
			if (m2 == nullptr) m2 = adam->M;
		}

		// if intersection point not found, return early so that no additional color is added (ie. background color is black)
		if (intersection_dist <= 0) return;

		// add emission light
		for (int channel = 0; channel < 3; channel++) image_data[channel] += m2->emissivity * m2->color[channel] * factor;

		// add reflected light

		// first, find reflection dir (using formula from https://www.scratchapixel.com/lessons/3d-basic-rendering/introduction-to-shading/reflection-refraction-fresnel.html)
		float view_reflect_dir[3];
		float temp = dot3D(local_view_dir, normal);
		for (int dim = 0; dim < 3; dim++) {
			view_reflect_dir[dim] = local_view_dir[dim] - 2 * temp * normal[dim];
		}
		normalize(view_reflect_dir, 3);

		// begin fresnel equations by finding theta_1 and theta_2 (snell's law)
		float intersection_eye_vector[3];
		for (int dim = 0; dim < 3; dim++) intersection_eye_vector[dim] = local_view_pos[dim] - intersection_point[dim];
		float theta_1 = getAngleVectors(normal, intersection_eye_vector);
		float theta_2 = asin(m1->refr_index * sin(theta_1) / m2->refr_index);

		float n1costheta1 = m1->refr_index * cos(theta_1);
		float n1costheta2 = m1->refr_index * cos(theta_2);
		float n2costheta1 = m2->refr_index * cos(theta_1);
		float n2costheta2 = m2->refr_index * cos(theta_2);

		// calculate reflection factor using fresnel equations
		float rs = (n1costheta1 - n2costheta2) / (n1costheta1 + n2costheta2);
		float rp = (n1costheta2 - n2costheta1) / (n1costheta2 + n2costheta1);
		float reflection_factor = (.5 * rs * rs + .5 * rp * rp);


		// next, get a random direction
		float rand_dir[3];
		randDirection(rand_dir, rand_seed);
		if (dot3D(rand_dir, normal) < 0) {
			for (int dim = 0; dim < 3; dim++) rand_dir[dim] *= -1;
		}

		// now, onto transmission
		float c1 = cos(theta_1);
		float c2 = sqrt(1 - power(m1->refr_index / m2->refr_index, 2) * power(sin(theta_1), 2));

		float view_transmit_dir[3];
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
		if (ray_depth == 0 && false) {
			image_data[0] = (normal[0] + 1) / 2 * 255;
			image_data[1] = 0 * (view_transmit_dir[1] + 1) / 2 * 255;
			image_data[2] = 0 * (view_transmit_dir[2] + 1) / 2 * 255;
			return;
		}

		// apply transmission OR reflection

		// remember to specify all variables used as input to findColor() function. define m1 and factor if they change! (factor should change!!!)
		if (randFloat(rand_seed) < reflection_factor) {
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

#endif


#ifdef GPU_ENABLE
void func(int WIDTH, int HEIGHT, unsigned char* image_data, float* image_data_float, int frames_still, float* eye_rotation, float* eye_pos, uint32_t* rand_seeds) {

	// emmission
	// optical density & roughness
	// refr index, general optical density, diffusion distance

	Material mats[4] = {
		Material(255, 255, 255, 0.0f, // air
				0, 0, 0, 0.0f,
				1.0f, 0.0f, .1f),
		Material(500, 500, 500, 1.0f, // light
				1, 1, 1, 0.0f,
				1.5f, 1.0f, .1f),
		Material(500, 0, 500, 0.0f, // light2
				.5f, 1, .5f, 0.0f,
				1.33f, .0f, 1.0f),
		Material(0, 0, 0, 0.0f, // wall
				1, .5f, 1, 1.0f,
				6.0f, 100.0f, .01f),
	};
	int air_mat = 0;
	
	/*
	Object objects[3] = {
		Object(.5f, 0, .5f, .1f, .1f, .1f, 1), // light 1
		Object(.2f, 0, .5f, .1f, 2), // purple orb
		Object(.5f, 0, .2f, .1f, 3), // green orb
	};
	int num_objects = 3;
	*/

	Object objects[4] = {
		Object(-2, 0, 0, 1, 1, 1, 3, 'B'), // wall
		Object(0, 2, 0, 1, 1, 1, 3, 'B'), // wall
		Object(0, 0, 2, 1, 1, 1, 3, 'B'), // wall
		Object(2, 0, -2, 1, 1, 1, 2, 'S'),
	};
	int num_objects = 4;

    sycl::buffer<unsigned char, 1> imageBuffer(image_data, sycl::range<1>(WIDTH * HEIGHT * 3));
	sycl::buffer<float, 1> imageBuffer_float(image_data_float, sycl::range<1>(WIDTH * HEIGHT * 3));
	sycl::buffer<float, 1> eye_rotation_buffer(eye_rotation, sycl::range<1>(2));
	sycl::buffer<float, 1> eye_position_buffer(eye_pos, sycl::range<1>(3));
	sycl::buffer<Object, 1> sphere_buffer(objects, sycl::range<1>(num_objects));
	sycl::buffer<Material, 1> mats_buffer(mats, sycl::range<1>(4));
	sycl::buffer<uint32_t, 1> random_buffer(rand_seeds, sycl::range<1>(WIDTH * HEIGHT));

	try {
		Q.submit([&](sycl::handler& h) {
			auto image = imageBuffer.get_access<sycl::access::mode::read_write>(h);
			auto image_float = imageBuffer_float.get_access<sycl::access::mode::read_write>(h);
			sycl::accessor<float> eye_position_acc = eye_position_buffer.get_access<sycl::access::mode::read_write>(h);
			sycl::accessor<float> eye_rotation_acc = eye_rotation_buffer.get_access<sycl::access::mode::read_write>(h);
			sycl::accessor<Object> sphere_acc = sphere_buffer.get_access<sycl::access::mode::read_write>(h);
			sycl::accessor<Material> mats_acc = mats_buffer.get_access<sycl::access::mode::read_write>(h);
			sycl::accessor<uint32_t> random_acc = random_buffer.get_access<sycl::access::mode::read_write>(h);

			h.parallel_for(sycl::range<1>(WIDTH * HEIGHT), [=](sycl::id<1> idx) {
				int w = idx % WIDTH;
				int h = idx / WIDTH;

				// calculate this fragment's direction
				float frag_dir[3] = { 2.0f * w / WIDTH - 1, 2.0f * h / HEIGHT - 1, 1.0f };
				rotateX(frag_dir, eye_rotation_acc[0]);
				rotateY(frag_dir, eye_rotation_acc[1]);
				normalize(frag_dir, 3);

				// call raytracing function to get this fragment's color
				float frag_color[3] = { 0, 0, 0 };
				float factors[3] = { 1, 1, 1 };
				// object array, num_objects, mats array, view_dir, eye_pos, factors, image_data, rand seeds, air_mat index
				findColor(&sphere_acc[0], num_objects, &mats_acc[0], frag_dir, &eye_position_acc[0], factors, frag_color, &random_acc[idx], air_mat);

				for (int channel = 0; channel < 3; channel++) {
					image_float[(h * WIDTH + w) * 3 + channel] *= frames_still / (frames_still + 1.0f);
					image_float[(h * WIDTH + w) * 3 + channel] += frag_color[channel] / (frames_still + 1.0f);
					image[(h * WIDTH + w) * 3 + channel] = std::min((int)image_float[(h * WIDTH + w) * 3 + channel], 255);
				}

				});
			}).wait();

	}
	catch (sycl::exception& e) {
		// Do something to output or handle the exception
		debug_file << e.what();
		debug_file.close();
		return;
	}

    
}
#else
void func(int WIDTH, int HEIGHT, unsigned char* image_data, float* image_data_float, int frames_still, float* eye_rotation, float* eye_pos, uint32_t* rand_seeds) {
	
	Object adam_objects[3] = {
		Object(.1f, .3f, 0, 3.5f, &light, .1f, .1f, .1f), // Box
		Object(.1f, .3f, .3f, 3.5f, &light2, .1f, .1f, .1f), // Box
		Object(.1f, .3f, 0, 1, .2f, &wall) // Object
	};

	Object adam(
		(Object*)adam_objects, 3,
		0, // combine method
		100, // bounding sphere radius
		0, 0, 0, // bounding sphere position
		&air // material index
	);

	// for each fragment (each pixel on screen)...
	for (int pixel_index = 0; pixel_index < WIDTH * HEIGHT; pixel_index++) {
		int w = pixel_index % WIDTH;
		int h = pixel_index / WIDTH;
		// calculate this fragment's direction
		float frag_dir[3] = { 2.0f * w / WIDTH - 1, 2.0f * h / HEIGHT - 1, 1 };
		rotateX(frag_dir, eye_rotation[0]);
		rotateY(frag_dir, eye_rotation[1]);
		normalize(frag_dir, 3);

		// call raytracing function to get this fragment's color
		float frag_color[3] = { 0, 0, 0 };
		findColor(&adam, frag_dir, eye_pos, 1.0, frag_color, &(rand_seeds[pixel_index]));
		for (int channel = 0; channel < 3; channel++) {
			image_data_float[(h * WIDTH + w) * 3 + channel] *= frames_still / (frames_still + 1.0f);
			image_data_float[(h * WIDTH + w) * 3 + channel] += frag_color[channel] / (frames_still + 1.0f);
			image_data[(h * WIDTH + w) * 3 + channel] = std::min((int)image_data_float[(h * WIDTH + w) * 3 + channel], 255);
		}
	}

}
#endif

