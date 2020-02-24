/**
 * particle_filter.cpp

 * Author: Zexu Wang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;
using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  num_particles = 100;
  default_random_engine gen;
  // Normal distribution for x, y and theta
  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);
  //weights.resize(num_particles);
  particles.resize(num_particles);
    for(auto& p: particles){
        p.x = dist_x(gen);
        p.y = dist_y(gen);
        p.theta = dist_theta(gen);
        p.weight = 1.0;
    }
is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {

  default_random_engine gen;
  normal_distribution<double> dist_x(0, std_pos[0]);
  normal_distribution<double> dist_y(0, std_pos[1]);
  normal_distribution<double> dist_theta(0, std_pos[2]);
  

  double dt = delta_t;
    for(auto& p: particles){
        if(fabs(yaw_rate)<0.001){
            p.x += velocity*dt*cos(p.theta);
            p.y += velocity*dt*sin(p.theta);
        }else{
            p.x += (velocity/yaw_rate)*(sin(p.theta+yaw_rate*dt)-sin(p.theta));
            p.y += (velocity/yaw_rate)*(cos(p.theta)-cos(p.theta+yaw_rate*dt));
            p.theta += yaw_rate*dt;
        }
        p.x += dist_x(gen);
        p.y += dist_y(gen);
        p.theta += dist_theta(gen);
    }

}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
    for(LandmarkObs& obs: observations){
        double near_nei = std::numeric_limits<float>::max();// Initialize the nearest distance by assign a large number to near_nei
        for(LandmarkObs& pred: predicted){
            double distance = dist(obs.x,obs.y,pred.x,pred.y);
            if(near_nei > distance){
                near_nei = distance;
                obs.id = pred.id;
            }
        }
    }

  /*for (unsigned int i = 0; i < observations.size(); ++i){
    double near_nei = std::numeric_limits<float>::max();// Initialize the nearest distance by assign a large number to near_nei
    for (unsigned int j = 0; j < predicted.size();++j){
      double distance = dist(observations[i].x,observations[i].y,predicted[j].x,predicted[j].y);
      if(near_nei > distance){
        near_nei = distance;
        observations[i].id = predicted[j].id;
      }
    }  
  }*/
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {

    for (auto& particle : particles) {
        particle.associations.clear();
        particle.sense_x.clear();
        particle.sense_y.clear();
        
        // Step 0, find all the landmarks within the sensor range for each particle
        std::vector<LandmarkObs> predictions;
        for (auto& landmark : map_landmarks.landmark_list) {
            double distance = dist(landmark.x_f, landmark.y_f, particle.x, particle.y);
            if (distance <= sensor_range) {
                predictions.push_back({landmark.id_i, landmark.x_f, landmark.y_f});
            }
        }
        
        // Step 1, transform the observations from vehicle coord to map coord
        std::vector<LandmarkObs> map_observations;
        for (auto& obs : observations) {
            double map_x = particle.x + obs.x * std::cos(particle.theta) - obs.y * std::sin(particle.theta);
            double map_y = particle.y + obs.x * std::sin(particle.theta) + obs.y * std::cos(particle.theta);
            map_observations.push_back({obs.id, map_x, map_y});
        }
        
        // Step 2, set association between the observations and the predictions
        dataAssociation(predictions, map_observations);
        
        // Step 3, calculate the particle's final weight
        particle.weight = 1.;
        double sig_x = std_landmark[0];
        double sig_y = std_landmark[1];
        double gauss_norm;
        gauss_norm = 1/(2*M_PI*sig_x*sig_y);
        for (const LandmarkObs& obs : map_observations) {
            for (const LandmarkObs& prd : predictions) {
                if (prd.id == obs.id) {
                    const double x_exp = std::exp(-pow(obs.x - prd.x, 2) / (2 * pow(sig_x, 2)));
                    const double y_exp = std::exp(-pow(obs.y - prd.y, 2) / (2 * pow(sig_x, 2)));
                    particle.weight *= gauss_norm * x_exp * y_exp;
                    particle.associations.push_back(prd.id);
                    particle.sense_x.push_back(prd.x);
                    particle.sense_y.push_back(prd.y);
                    break;
                }
            }
        }
    }
    
}

void ParticleFilter::resample() {
    std::vector<double> weights(num_particles, 1.0);
    for(int i = 0; i < num_particles; ++i){
        weights[i] = particles[i].weight;
    }
  vector<Particle> new_particles;// New blank particles vector 
  new_particles.resize(num_particles);
  // generate random number according to its weight
  random_device rd;
  default_random_engine gen(rd());
  for (int i = 0; i < num_particles; ++ i){
    discrete_distribution<int> index(weights.begin(), weights.end());
    new_particles[i] = particles[index(gen)];
  }
  particles = new_particles;
  
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
 
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}
