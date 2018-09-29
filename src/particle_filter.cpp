/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
    // set number of particles

    num_particles = 50; //1000, 200, 100, 20

    // initialize gaussian noises
    normal_distribution<double> dist_x(x, std[0]);
    normal_distribution<double> dist_y(y, std[1]);
    normal_distribution<double> dist_theta(theta, std[2]);

    // initialize particles
    for (int i = 0; i < num_particles; i++){
        Particle p;
        p.id = i;
        p.weight = 1.0;

        // add gaussian noises
        p.x = dist_x(gen);
        p.y = dist_y(gen);
        p.theta = dist_theta(gen);

        particles.push_back(p);
        weights.push_back(p.weight);
    }

    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	// set gaussian noises
	normal_distribution<double> dist_x_pred(0, std_pos[0]), dist_y_pred(0, std_pos[1]), dist_theta_pred(0, std_pos[2]);

	for (int i=0; i < num_particles; i++){

	    double x = particles[i].x, y = particles[i].y, theta = particles[i].theta;

	    if (yaw_rate == 0){
	        x += velocity*delta_t*cos(theta);
            y += velocity*delta_t*sin(theta);

	    }
	    else{
            x += velocity/yaw_rate*(sin(theta+yaw_rate*delta_t)-sin(theta));
            y += velocity/yaw_rate*(cos(theta)-cos(theta+yaw_rate*delta_t));
            theta += yaw_rate*delta_t;
	    }

	    // add guassian noise
        x += dist_x_pred(gen);
        y += dist_y_pred(gen);
        theta += dist_theta_pred(gen);

	    particles[i].x = x;
	    particles[i].y = y;
	    particles[i].theta = theta;

	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
    // find the nearest neighbour

    for (int i = 0; i <observations.size(); i++){
        LandmarkObs observe = observations[i];

        // initialize landmark id
        int landmark_id;
        // set the maximum possible distance as the minimum
        double min_distance  = numeric_limits<double>::max();

        for (int j = 0; j < predicted.size(); j++){
            LandmarkObs pred = predicted[j];
            double distance = dist(observe.x, observe.y, pred.x, pred.y);
            if (distance < min_distance){
                min_distance = distance;
                landmark_id = pred.id;
            }
        }

        // set the nearest landmark id as the observation landmark id
        observations[i].id = landmark_id;
    }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

    for (int i =  0; i < num_particles; i++){
        double x = particles[i].x, y = particles[i].y, theta = particles[i].theta;

        // A vector of the locations of predicted landmarks
        vector<LandmarkObs> pred_lm;

        for (int j = 0; j < map_landmarks.landmark_list.size(); j++){
            int lm_id = map_landmarks.landmark_list[j].id_i;
            double lm_x = map_landmarks.landmark_list[j].x_f, lm_y = map_landmarks.landmark_list[j].y_f;
            LandmarkObs land_m = {lm_id, lm_x, lm_y};

            // select landmarks within sensor range
            if (fabs(dist(lm_x, lm_y, x, y)) <= sensor_range){
                pred_lm.push_back(land_m);
            }
        }

        // transformation
        vector<LandmarkObs> transformed_obs;

        for (int j = 0; j < observations.size(); j++){
            double trans_x = x + observations[j].x*cos(theta) - observations[j].y*sin(theta);
            double trans_y = y + observations[j].x*sin(theta) + observations[j].y*cos(theta);

            LandmarkObs transformed_obj;
            transformed_obj = {observations[j].id, trans_x, trans_y};
            transformed_obs.push_back(transformed_obj);

        }

        dataAssociation(pred_lm, transformed_obs);

        particles[i].weight = 1.0;
        weights[i] = 1.0;

        for (int m = 0; m < transformed_obs.size(); m++){
            int obs_id = transformed_obs[m].id;
            double obs_x = transformed_obs[m].x, obs_y = transformed_obs[m].y;
            double pred_x, pred_y;

            for (int n = 0; n < pred_lm.size(); n++){
                if (pred_lm[n].id == obs_id){
                    pred_x = pred_lm[n].x;
                    pred_y = pred_lm[n].y;
                }
            }

            double w = (1/(2*M_PI*std_landmark[0]*std_landmark[1])) * exp(-(pow(pred_x-obs_x, 2)/(2*pow(std_landmark[0],2)) +
                       pow(pred_y-obs_y, 2)/(2*pow(std_landmark[1],2)) - (2*(pred_x-obs_x)*(pred_y-obs_y)/(sqrt(std_landmark[0])*sqrt(std_landmark[1])))));

            particles[i].weight *= w;
            weights[i] = w;
        }
    }
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	discrete_distribution<int> index(weights.begin(), weights.end());

	vector<Particle> resampled_particles;

	for (int i = 0; i < num_particles; i++){
	    Particle p = particles[index(gen)];
	    resampled_particles.push_back(p);
	}
    particles = resampled_particles;

}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates
/*
    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
*/

    particle.associations.clear();
    particle.sense_x.clear();
    particle.sense_y.clear();

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;

    return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
