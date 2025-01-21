classdef RISISAC_V2X_Sim < handle
    properties
        % System parameters
        fc = 28e9                 % Carrier frequency (28 GHz)
        B = 20e6                  % Bandwidth (20 MHz)
        Ns = 10                   % Number of subcarriers
        Nb = 4                    % Number of BS antennas
        Nt = 4                    % Number of target antennas
        Nr = 64                   % Number of RIS elements
        
        % Path loss parameters
        alpha_l = 3.2             % Direct path loss exponent
        alpha_nl = 2.2            % RIS path loss exponent
        sigma_l = 3               % Direct path shadow fading
        sigma_nl = 4              % RIS path shadow fading
        
        % Locations (in meters)
        bs_loc = [900, 100, 20]   % Base station location
        ris_loc = [200, 300, 40]  % RIS location
        target_loc = [500, 500, 0]% Target location
        
        % Environment dimensions
        env_dims = [1000, 1000]   % Environment dimensions
        
        % Channel matrices
        H_d      % Direct channel
        H_br     % BS-RIS channel
        H_rt     % RIS-target channel
        Phi      % RIS phase shifts
        
        % Vehicle dynamics parameters
        vehicle = struct(...
            'position', [0, 0, 0], ...          % [x, y, z] in meters
            'velocity', [0, 0, 0], ...          % [vx, vy, vz] in m/s
            'acceleration', [0, 0, 0], ...      % [ax, ay, az] in m/s^2
            'heading', 0, ...                   % in radians
            'mass', 1500, ...                   % in kg
            'max_speed', 30, ...                % in m/s
            'max_acceleration', 3, ...          % in m/s^2
            'max_deceleration', -6, ...         % in m/s^2
            'max_steering_angle', pi/4, ...     % in radians
            'wheelbase', 2.7 ...                % in meters
        )
        
        % Simulation parameters
        dt = 0.1  % Time step in seconds
    end
    
    methods
        function obj = RISISAC_V2X_Sim()
            % Initialize channels
            obj.initializeChannels();
        end
        
        function initializeChannels(obj)
            % Initialize direct channel
            obj.H_d = obj.generateDirectChannel();
            
            % Initialize BS-RIS channel
            obj.H_br = obj.generateBSRISChannel();
            
            % Initialize RIS-target channel
            obj.H_rt = obj.generateRISTargetChannel();
            
            % Initialize RIS phase shifts
            obj.Phi = eye(obj.Nr);
        end
        
        function H = generateDirectChannel(obj)
            % Generate direct channel with path loss and shadow fading
            d = norm(obj.bs_loc - obj.target_loc);
            PL = 20*log10(4*pi*d*obj.fc/3e8) + 10*obj.alpha_l*log10(d);
            shadow = obj.sigma_l * randn(1);
            
            % Generate Rician fading channel
            K_factor = 10;  % Rician K-factor
            H = sqrt(1/(1+K_factor)) * (sqrt(1/2)*(randn(obj.Nt,obj.Nb) + 1j*randn(obj.Nt,obj.Nb))) + ...
                sqrt(K_factor/(1+K_factor)) * ones(obj.Nt,obj.Nb);
            
            % Apply path loss and shadow fading
            H = H * 10^(-(PL + shadow)/20);
        end
        
        function H = generateBSRISChannel(obj)
            % Generate BS-RIS channel with path loss
            d = norm(obj.bs_loc - obj.ris_loc);
            PL = 20*log10(4*pi*d*obj.fc/3e8) + 10*obj.alpha_nl*log10(d);
            shadow = obj.sigma_nl * randn(1);
            
            % Generate channel matrix
            H = sqrt(1/2)*(randn(obj.Nr,obj.Nb) + 1j*randn(obj.Nr,obj.Nb));
            
            % Apply path loss and shadow fading
            H = H * 10^(-(PL + shadow)/20);
        end
        
        function H = generateRISTargetChannel(obj)
            % Generate RIS-target channel with path loss
            d = norm(obj.ris_loc - obj.target_loc);
            PL = 20*log10(4*pi*d*obj.fc/3e8) + 10*obj.alpha_nl*log10(d);
            shadow = obj.sigma_nl * randn(1);
            
            % Generate channel matrix
            H = sqrt(1/2)*(randn(obj.Nt,obj.Nr) + 1j*randn(obj.Nt,obj.Nr));
            
            % Apply path loss and shadow fading
            H = H * 10^(-(PL + shadow)/20);
        end
        
        function updateVehicleDynamics(obj, throttle, steering_angle)
            % Constrain inputs
            throttle = max(min(throttle, 1), -1);  % Normalize between -1 and 1
            steering_angle = max(min(steering_angle, obj.vehicle.max_steering_angle), ...
                               -obj.vehicle.max_steering_angle);
            
            % Calculate acceleration based on throttle
            if throttle >= 0
                target_accel = throttle * obj.vehicle.max_acceleration;
            else
                target_accel = throttle * abs(obj.vehicle.max_deceleration);
            end
            
            % Update heading based on steering angle and velocity
            speed = norm(obj.vehicle.velocity(1:2));
            if speed > 0.1  % Only update heading when moving
                turning_radius = obj.vehicle.wheelbase / tan(steering_angle);
                angular_velocity = speed / turning_radius;
                obj.vehicle.heading = obj.vehicle.heading + angular_velocity * obj.dt;
                % Normalize heading to [-pi, pi]
                obj.vehicle.heading = mod(obj.vehicle.heading + pi, 2*pi) - pi;
            end
            
            % Update acceleration, velocity, and position
            % Decompose acceleration into x and y components based on heading
            obj.vehicle.acceleration(1) = target_accel * cos(obj.vehicle.heading);
            obj.vehicle.acceleration(2) = target_accel * sin(obj.vehicle.heading);
            
            % Update velocity using acceleration
            new_velocity = obj.vehicle.velocity + obj.vehicle.acceleration * obj.dt;
            
            % Constrain speed to maximum
            current_speed = norm(new_velocity(1:2));
            if current_speed > obj.vehicle.max_speed
                new_velocity(1:2) = new_velocity(1:2) * obj.vehicle.max_speed / current_speed;
            end
            obj.vehicle.velocity = new_velocity;
            
            % Update position
            obj.vehicle.position = obj.vehicle.position + obj.vehicle.velocity * obj.dt;
        end
        
        function state = getState(obj)
            % Construct state vector for RL agent
            % Format: [position(3), velocity(3), acceleration(3), heading(1), 
            %          ris_phases_real(64), ris_phases_imag(64)]
            state = [
                obj.vehicle.position(:)', ...           % 3 elements
                obj.vehicle.velocity(:)', ...           % 3 elements
                obj.vehicle.acceleration(:)', ...       % 3 elements
                obj.vehicle.heading, ...                % 1 element
                real(diag(obj.Phi))', ...              % 64 elements
                imag(diag(obj.Phi))'                   % 64 elements
            ];
            
            % Ensure it's a row vector
            state = state(:)';
        end
        
        function [next_state, reward, done] = step(obj, action)
            % Parse action vector
            ris_phases = action(1:obj.Nr);           % RIS phase shifts
            throttle = action(obj.Nr + 1);           % Vehicle throttle
            steering = action(obj.Nr + 2);           % Steering angle
            
            % Update RIS phase shifts
            obj.Phi = diag(exp(1j * 2 * pi * ris_phases));
            
            % Update vehicle state
            obj.updateVehicleDynamics(throttle, steering);
            
            % Calculate performance metrics
            precoder = eye(obj.Nb) / sqrt(obj.Nb);  % Simple precoder
            [rate, peb] = obj.calculatePerformanceMetrics(precoder);
            
            % Calculate reward
            reward = obj.calculateReward(rate, peb);
            
            % Get next state
            next_state = obj.getState();
            
            % Check if episode is done
            done = obj.isEpisodeDone();
        end
        
        function [rate, peb] = calculatePerformanceMetrics(obj, precoder)
            % Calculate achievable rate
            H_eff = obj.H_d + obj.H_rt * obj.Phi * obj.H_br;
            SNR = 20; % dB
            rate = log2(det(eye(obj.Nt) + 10^(SNR/10) * H_eff * (precoder * precoder') * H_eff'));
            
            % Calculate Position Error Bound (PEB)
            F = zeros(3,3); % Fisher Information Matrix
            for k = 1:obj.Ns
                H_k = H_eff; % Consider frequency-dependent channel
                J_k = H_k * (precoder * precoder') * H_k';
                F = F + real(J_k);
            end
            peb = sqrt(trace(inv(F)));
        end
        
        function reward = calculateReward(obj, rate, peb)
            % Calculate reward based on rate, PEB, and vehicle state
            w1 = 0.6;  % Weight for rate
            w2 = 0.3;  % Weight for PEB
            w3 = 0.1;  % Weight for vehicle dynamics
            
            % Normalize metrics
            norm_rate = rate / 10;  % Assuming max rate is 10 bps/Hz
            norm_peb = min(1, peb / 10);  % Assuming max PEB is 10m
            
            % Calculate vehicle dynamics component
            speed = norm(obj.vehicle.velocity(1:2));
            speed_reward = speed / obj.vehicle.max_speed;  % Reward for maintaining good speed
            
            reward = w1 * norm_rate - w2 * norm_peb + w3 * speed_reward;
        end
        
        function done = isEpisodeDone(obj)
            % Check if vehicle is out of bounds
            pos = obj.vehicle.position;
            done = any(pos(1:2) < 0) || ...
                   pos(1) > obj.env_dims(1) || ...
                   pos(2) > obj.env_dims(2);
        end
        
        function state = reset(obj)
            % Reset simulation state
            obj.initializeChannels();
            
            % Reset vehicle state
            obj.vehicle.position = [0, 0, 0];
            obj.vehicle.velocity = [0, 0, 0];
            obj.vehicle.acceleration = [0, 0, 0];
            obj.vehicle.heading = 0;
            
            % Return the current state
            state = obj.getState();
            
        end

    end
end