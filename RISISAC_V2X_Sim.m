% import java.math.*
classdef RISISAC_V2X_Sim < handle
    properties
        % System parameters
        fc = 28e9                 % Carrier frequency (28 GHz)
        B = 20e6                  % Bandwidth (20 MHz)
        Ns = 10                   % Number of subcarriers
        Nb = 4                    % Number of BS antennas
        Nt = 4                    % Number of target antennas
        Nr = 64                   % Number of RIS elements
        Nx = 8
        Ny = 8
        Mb = 64
        rate = 0
        c = 3e8;
        lambda = 3e8/28e9;
        sigma_c_sq = 1.38e-23 * 290 * 20e6; 

        h_l = 0;
        h_nl = 0;
        gamma_c = 0;
        SNR = 0;
        W
        Wx
        noise_var = 0;
        cc = 0;
        speed = 10
        end_x = 1000
        Pb = 0
        R_min = 0
        H_combined
        Jzao
        waypoint_idx = 1;

        currentWaypointIndex = 1;   % Start from the first waypoint
        reachedWaypoint = true;     % Ensure path planning starts immediately
        currentPath = [];           % No path computed initially
        pathProgress = 0;           % No progress on the path initially
        minTurningRadius = 5;       % Minimum turning radius for Dubins path

        center = [250,250,0]; % Center of circular path
        radius = 250;
        angular_speed = 0.04;
        angle

        % Define waypoints: [x, y, theta]
        waypoints = [
            0, 0, 0;
            1000, 0, 0;
            1000, 1000, 0;
            0, 1000, 0;
            500, 500, 0
        ];


        % Path loss parameters
        alpha_l = 3.2             % Direct path loss exponent
        alpha_nl = 2.2            % RIS path loss exponent
        rho_l = 3               % Direct path shadow fading
        rho_nl = 4              % RIS path shadow fading
        
        % starting_pos = [500,500,0]
        % Locations (in meters)
        target_loc = [500, 500, 0]% Target location

        stepCount = 0;
        maxSteps = 10000;
        
        car_loc = [500,500,0];
        % Environment dimensions
        % env_dims = [1000, 1000]   % Environment dimensions
        destination;
        time = 0;
        arrival_threshold = 10
        car_orientation = 0
        current_speed = 10
        max_speed = 30
        acceleration = 2
        integral_error = 0
        prev_error = 0

        minpeb = 10000;
        peb = 0;

        visualizationAxes;
        trajectory = [];

        % Channel matrices
        H_bt      % Direct channel
        H_br     % BS-RIS channel
        H_rt     % RIS-target channel
        
        phi      % RIS phase shifts
        % Simulation parameters
        dt = 0.1  % Time step in seconds
    end

    properties (Constant)
        bs_loc = [900, 100, 20];   % Base station location
        ris_loc = [200, 300, 40];  % RIS location
        starting_pos = [500, 500, 0]; % Starting position of the vehicle
        env_dims = [1000, 1000] 
    end
    methods (Access = private)
        function initializeEpisode(obj)
            fprintf("Initializing Episode...\n");
            obj.center = [250, 250, 0]; % Center of circular path
            obj.radius = 250; % Radius of circular path in meters
            obj.angular_speed = 0.04; % Angular speed in radians per second
            obj.car_loc = [500, 500, 0];
        
            % Force correct starting angle (45 degrees for quadrant I motion)
            obj.angle = atan2(obj.car_loc(2) - obj.center(2), obj.car_loc(1) - obj.center(1));
        
            % Set initial orientation tangent to the circle
            obj.car_orientation = obj.angle + pi/2;
        
            % Debugging statement
            fprintf("Initial Car Location: (%.2f, %.2f)\n", obj.car_loc(1), obj.car_loc(2));
        end        
    end       
    
    methods
        function obj = RISISAC_V2X_Sim()
            % Initialize channels
            obj.initializeChannels();
            obj.initializephi();
            obj.calculate_pathloss();
            obj.destination = [randi([0, 1000]), randi([0, 1000]), 0];
            obj.initializeVisualization();
            % obj.destination = [999, 999, 0];
            obj.H_combined = obj.compute_Heff();
            [obj.Wx,obj.W] = computeWx(obj);
            obj.Pb = obj.getpower();
            obj.gamma_c = computeSNR(obj);
            obj.rate = obj.getrate();
            obj.peb = obj.calculatePerformanceMetrics(obj.Wx);
        end

        function nb = get_Nb(obj)
            nb = obj.Nb;
        end

        function calculate_pathloss(obj)
            K_dB = 4; % Rician K-factor in dB
            K = 10^(K_dB/10);
            sigma = sqrt(1/(2*(K+1)));
            mu = sqrt(K/(K+1));
            obj.h_l = (sigma * complex(randn(1,1), randn(1,1))) + mu;
            obj.h_nl = (sigma * complex(randn(1,1), randn(1,1))) + mu;
        end

        function H_combined = compute_Heff(obj)
            [HLos,HLos_3d] = generate_H_Los(obj, obj.H_bt, obj.Nt, obj.Nb);
            [HNLos,HNLos_3d] = generate_H_NLoS(obj, obj.H_rt, obj.H_br, obj.Nt, obj.Nr, obj.Nb);
            H_combined = HLos + HNLos;
            obj.H_combined = H_combined;
        end

        function gamma_c = computeSNR(obj)
            [~,HLos_3d] = generate_H_Los(obj, obj.H_bt, obj.Nt, obj.Nb);
            [~,HNLos_3d] = generate_H_NLoS(obj, obj.H_rt, obj.H_br, obj.Nt, obj.Nr, obj.Nb);
            obj.Pb = getpower(obj);
            gamma_c_per_subcarrier = zeros(1, obj.Ns);
            for n = 1:obj.Ns
                H_combined_n = HLos_3d(:,:,n) + HNLos_3d(:,:,n);
                gamma_c_per_subcarrier(n) = obj.Pb * norm(H_combined_n * obj.W, 'fro')^2 / obj.sigma_c_sq;
            end
            obj.gamma_c = obj.Ns / sum(1 ./ gamma_c_per_subcarrier);  
            obj.SNR = 10 * log10(obj.gamma_c);
            gamma_c = obj.gamma_c;
        end

        function Pb = getpower(obj) 
            Pb = trace(obj.Wx * obj.Wx') / obj.Ns;
            obj.Pb = Pb;
        end

        function rate = getrate(obj)
            rate = obj.B*log2(1+obj.gamma_c);
            obj.rate = rate;
        end

        % ! -------------------- CHANNEL INITIALIZATION PART STARTS HERE --------------------        
        function initializeChannels(obj)
            [obj.H_bt, obj.H_br, obj.H_rt] = generate_channels(obj, obj.Nt, obj.Nr, obj.Nb);   
            obj.H_combined = compute_Heff(obj);
            
        end

        function initializephi(obj)
            rho_r = 1;
            Bit = 2;  
            Delta_delta = 2*pi / (2^Bit);  
            A = (0 : (2^Bit - 1)) * Delta_delta;
            theta = A(randi(numel(A), obj.Nr, 1));
            u = rho_r * exp(1j*theta);
            obj.phi = diag(u); 
        end

        
        function [H_bt, H_br, H_rt] = generate_channels(obj, Nt, Nr, Nb)
            % Constants
            d = obj.lambda/2;
            dr = obj.lambda/2;
            [~,~,~,~,~,~,~,angles] = obj.computeGeometricParameters();
            
            H_bt = generate_H_bt(obj, Nt, Nb, angles, obj.lambda, d);
            [H_br, H_rt] = generate_H_br_H_rt(obj, Nb, Nr, Nt, angles, obj.lambda, d, dr);
        end

        function H_bt = generate_H_bt(obj, Nt, Nb, angles, lambda, d)
            psi_bt = angles.bs_to_target_transmit;
            psi_tb = angles.bs_to_target_receive;
            
            % Get frequency-dependent steering vectors (dimensions: Nb×Ns and Nt×Ns)
            a_psi_bt = compute_a_psi(obj, Nb, psi_bt, lambda, d);
            a_psi_tb = compute_a_psi(obj, Nt, psi_tb, lambda, d);
            
            % Initialize 3D channel matrix (Nt×Nb×Ns)
            H_bt = zeros(Nt, Nb, obj.Ns);
            
            % Calculate H_bt for each subcarrier
            for n = 1:obj.Ns
                % Outer product of steering vectors for nth subcarrier
                H_bt(:,:,n) = a_psi_tb(:,n) * a_psi_bt(:,n)';
            end
        end
        

        function [H_br, H_rt] = generate_H_br_H_rt(obj, Nb, Nr, Nt, angles, lambda, d, dr)
            % BS-RIS channel parameters
            psi_br = angles.bs_to_ris.azimuth;
            phi_abr = angles.bs_to_ris.elevation_azimuth;
            phi_ebr = angles.bs_to_ris.elevation_angle;
            
            % RIS-Target channel parameters
            psi_rt = angles.ris_to_target.azimuth;
            phi_art = angles.ris_to_target.elevation_angle;
            phi_ert = angles.ris_to_target.elevation_angle;
            
            % Generate channels
            H_br = generate_H_br(obj, Nr, Nb, phi_abr, phi_ebr, psi_br, lambda, dr, d);
            H_rt = generate_H_rt(obj, Nt, Nr, phi_art, phi_ert, psi_rt, lambda, dr, d);
        end

        function H_br = generate_H_br(obj, Nr, Nb, phi_abr, phi_ebr, psi_br, lambda, dr, d)
            % Get frequency-dependent steering vectors (dimensions: Nb×Ns and Nr×Ns)
            a_psi_br = compute_a_psi(obj, Nb, psi_br, lambda, d);
            a_phi_abr = compute_a_phi(obj, sqrt(Nr), phi_abr, phi_ebr, lambda, dr);
            
            % Initialize 3D channel matrix (Nr×Nb×Ns)
            H_br = zeros(Nr, Nb, obj.Ns);
            
            % Calculate H_br for each subcarrier
            for n = 1:obj.Ns
                % Outer product of steering vectors for nth subcarrier
                H_br(:,:,n) = a_phi_abr(:,n) * a_psi_br(:,n)';
            end
        end
        
        
        function H_rt = generate_H_rt(obj, Nt, Nr, phi_art, phi_ert, psi_rt, lambda, dr, d)
            % Get frequency-dependent steering vectors (dimensions: Nt×Ns and Nr×Ns)
            a_psi_rt = compute_a_psi(obj, Nt, psi_rt, lambda, d);
            a_phi_art = compute_a_phi(obj, sqrt(Nr), phi_art, phi_ert, lambda, dr);
            
            % Initialize 3D channel matrix (Nt×Nr×Ns)
            H_rt = zeros(Nt, Nr, obj.Ns);
            
            % Calculate H_rt for each subcarrier
            for n = 1:obj.Ns
                % Outer product of steering vectors for nth subcarrier
                H_rt(:,:,n) = a_psi_rt(:,n) * a_phi_art(:,n)';
            end
        end

        function a_vec = compute_a_psi(obj, Nant, psi, lambda, d)
            Ns = obj.Ns; % Number of subcarriers
            B = obj.B;   % Bandwidth (Hz)
            fc = obj.fc; % Carrier frequency (Hz)
            
            n_ant = (0:(Nant-1)).'; % antenna indices (column vector)
            a_vec = zeros(Nant, Ns); % Initialize output matrix
            
            for n = 1:Ns
                % Frequency of nth subcarrier (assuming centered around fc)
                f_n = fc + B*(n - (Ns+1)/2)/Ns;
                k_n = 2*pi*f_n/(3e8); % wavenumber at frequency f_n
                
                phase_terms = exp(1j * k_n * d * n_ant * sin(psi));
                a_vec(:, n) = phase_terms / sqrt(Nant);
            end
        end

        function a_phi = compute_a_phi(obj, Nx, phi_a, phi_e, lambda, dr)
            Ns = obj.Ns; % Number of subcarriers
            B = obj.B;   % Bandwidth (Hz)
            fc = obj.fc; % Carrier frequency (Hz)
        
            N2 = Nx * Nx;
            a_phi = zeros(N2, Ns); % Initialize output matrix
        
            for n_subcarr = 1:Ns
                % Frequency of nth subcarrier (assuming centered around fc)
                f_n = fc + B*(n_subcarr - (Ns+1)/2)/Ns;
                k_n = 2*pi*f_n/(3e8); % wavenumber at frequency f_n
        
                idx = 1;
                for m_idx = 1:Nx
                    for n_idx = 1:Nx
                        phase_term = exp(1j * k_n * dr * ...
                            (m_idx * sin(phi_a) * sin(phi_e) + n_idx * cos(phi_e)));
                        a_phi(idx, n_subcarr) = phase_term;
                        idx = idx + 1;
                    end
                end
        
                % Normalize per subcarrier
                a_phi(:, n_subcarr) = a_phi(:, n_subcarr) / sqrt(N2);
            end
        end
        
        function [H_Los, H_Los_3d] = generate_H_Los(obj, H_bt, Nt, Nb)            
            gamma_l = sqrt(Nb*Nt)/sqrt(obj.rho_l);
            
            [~,~,~,~, ~, ~, delays, ~] = computeGeometricParameters(obj);
            tau_l = delays.line_of_sight;
            
            % Initialize 3D channel matrix (Nt×Nb×Ns)
            H_Los_3d = zeros(Nt, Nb, obj.Ns);
            
            % Calculate H_Los for each subcarrier
            for n = 1:obj.Ns
                % Calculate phase shift for nth subcarrier
                phase = exp(1j*2*pi*obj.B*(n-1)*tau_l/obj.Ns);
                
                % H_bt is now a 3D matrix, so we use H_bt(:,:,n)
                H_Los_3d(:,:,n) = gamma_l * obj.h_l * H_bt(:,:,n) * phase;
            end
        
            % Return both the 3D channel matrix and its average
            H_Los = mean(H_Los_3d, 3);  % Average across subcarriers (Nt×Nb matrix)
        end
        
        function [H_NLoS, H_NLoS_3d] = generate_H_NLoS(obj, H_rt, H_br, Nt, Nr, Nb)            
            obj.rho_nl = 4;
            
            gamma_nl = sqrt(Nb*Nr)/sqrt(obj.rho_nl);
            
            [~,~,~,~, ~, ~, delays, ~] = computeGeometricParameters(obj);
            tau_nl = delays.non_line_of_sight;
        
            % Generate RIS phase shift matrix
            rho_r = 1;
            theta = 2*pi*rand(Nr,1);
            u = rho_r * exp(1j*theta);
            obj.phi = diag(u);
            
            % Initialize 3D channel matrix (Nt×Nb×Ns)
            H_NLoS_3d = zeros(Nt, Nb, obj.Ns);
            
            % Calculate H_NLoS for each subcarrier
            for n = 1:obj.Ns
                phase = exp(1j*2*pi*obj.B*(n-1)*tau_nl/obj.Ns);
                H_NLoS_3d(:,:,n) = gamma_nl * obj.h_nl * H_rt(:,:,n) * obj.phi * H_br(:,:,n) * phase;
            end
            
            % Return both the 3D channel matrix and its average
            H_NLoS = mean(H_NLoS_3d, 3);  % Average across subcarriers (Nt×Nb matrix)
        end
               
        

        % ! -------------------- CHANNEL INITIALIZATION PART ENDS HERE --------------------        
        
        % ! -------------------- MACHINE LEARNING PART STARTS HERE --------------------        
        function state = getState(obj)
            phi_real = real(diag(obj.phi))';
            phi_imag = imag(diag(obj.phi))'; 
            Rc = obj.rate;
            
            H = obj.H_combined;
            H_real = real(H(:))';
            H_imag = imag(H(:))';
            state = [
                phi_real, ...
                phi_imag, ...
                Rc, ...
                H_real, ... 
                H_imag ...
            ];
            state = real(state);
            state = state(:)';
            % disp(size(state));
        end

        function initializeVisualization(obj)
            figure;
            obj.visualizationAxes = axes;
            hold(obj.visualizationAxes, 'on');
            grid(obj.visualizationAxes, 'on');
            xlabel(obj.visualizationAxes, 'X Position (m)');
            ylabel(obj.visualizationAxes, 'Y Position (m)');
            zlabel(obj.visualizationAxes, 'Z Position (m)');
            title(obj.visualizationAxes, '3D Trajectory of Vehicle with RIS and BS Locations');
            view(obj.visualizationAxes, 3);
            
            obj.trajectory = obj.car_loc;
        end

        function updateVisualization(obj)
            % Clear previous plot
            cla(obj.visualizationAxes);
        
            % Plot the vehicle's current location
            plot3(obj.visualizationAxes, obj.car_loc(1), obj.car_loc(2), obj.car_loc(3), 'bo', 'MarkerSize', 8, 'DisplayName', 'Vehicle');
        
            % Plot the RIS location
            plot3(obj.visualizationAxes, obj.ris_loc(1), obj.ris_loc(2), obj.ris_loc(3), 'rs', 'MarkerSize', 10, 'DisplayName', 'RIS');
        
            % Plot the BS location
            plot3(obj.visualizationAxes, obj.bs_loc(1), obj.bs_loc(2), obj.bs_loc(3), 'g^', 'MarkerSize', 10, 'DisplayName', 'Base Station');
        
            % Plot the trajectory of the vehicle
            plot3(obj.visualizationAxes, obj.trajectory(:,1), obj.trajectory(:,2), obj.trajectory(:,3), 'b-', 'DisplayName', 'Trajectory');
        
            % Update the legend
            legend(obj.visualizationAxes, 'show');
        
            % Pause to update the plot
            pause(0.01);
        end
        
        function updateTrajectory(obj)
            obj.trajectory = [obj.trajectory; obj.car_loc];
        end        
        

        function [next_state, reward, peb, rate, power, done] = step(obj, action)
            % Update RIS phases based on action
            ris_phases = action(1:obj.Nr);
            obj.phi = diag(exp(1j * 2 * pi * ris_phases));
            obj.H_combined = compute_Heff(obj);
            [obj.Wx,obj.W] = computeWx(obj);
            power = getpower(obj);
            obj.Pb = power;
            obj.gamma_c = computeSNR(obj);
            rate = getrate(obj);
            obj.rate = rate;
            peb = obj.calculatePerformanceMetrics(obj.Wx);
            obj.peb = peb;
            % Calculate performance metrics
            reward = obj.computeReward(peb);
            reward = abs(reward); % Ensure reward is a real-valued scalar
            
            % Initialize properties only once per episode
            persistent isInitialized;
            if isempty(isInitialized) || obj.stepCount == 0
                isInitialized = true;
                obj.center = [250, 500, 0]; % Center of circular path
                obj.radius = 250; % Radius of circular path in meters
                obj.angular_speed = 0.08; % Angular speed in radians per second
                obj.angle = 0; % Initial angle
                % Force car to start at a position on the circle
                obj.car_loc = obj.center + [obj.radius, 0, 0];
                obj.car_orientation = pi/2; % Start with orientation tangent to circle
            end
            
            % Update angle for circular motion
            obj.angle = obj.angle + obj.angular_speed * obj.dt;
            
            % Calculate new position on the circle
            new_x = obj.center(1) + obj.radius * cos(obj.angle);
            new_y = obj.center(2) + obj.radius * sin(obj.angle);
            obj.car_loc = [new_x, new_y, 0];
            
            % Update car orientation to be tangent to the circle
            obj.car_orientation = obj.angle + pi/2;
            
            % Update target location to match car location
            obj.target_loc = obj.car_loc;
            
            % Update time and step count
            obj.time = obj.time + obj.dt;
            obj.stepCount = obj.stepCount + 1;
            
            % Update trajectory and visualization
            obj.updateTrajectory();
            obj.updateVisualization();
            
            % Check if episode is done
            done = obj.stepCount >= obj.maxSteps;
            
            % Retrieve the next state
            next_state = getState(obj);
        end
                

        % function [next_state, reward, peb, rate, power, done] = step(obj, action)
        %     % Update RIS phases based on action
        %     ris_phases = action(1:obj.Nr);
        %     obj.phi = diag(exp(1j * 2 * pi * ris_phases));
            
        %     % Calculate performance metrics
        %     peb = obj.calculatePerformanceMetrics();
        %     rate = getrate(obj);
        %     power = getpower(obj);
        %     reward = obj.computeReward(peb);
        %     reward = abs(reward); % Ensure reward is a real-valued scalar
            
        %     % Vehicle dynamics parameters
        %     max_speed = 30; % Maximum speed in m/s
        %     max_acceleration = 2; % Maximum acceleration in m/s^2
        %     max_turning_angle = pi/6; % Maximum turning angle in radians
            
        %     % Initialize properties only once per episode
        %     % Create a persistent variable to track initialization status
        %     persistent isInitialized;
        %     if isempty(isInitialized) || obj.stepCount == 0
        %         isInitialized = true;
        %         obj.waypoint_idx = 1;
        %         obj.destination = [1000, 500, 0]; % First destination
        %         obj.integral_error = 0;
        %         obj.prev_error = 0;
        %         obj.current_speed = 10; % Start with a non-zero speed
        %         obj.car_orientation = 0; % Start facing along the x-axis
        %         % Force car to start at starting_pos at beginning of episode
        %         obj.car_loc = obj.starting_pos;
        %     end
            
        %     % Define waypoint for current index
        %     switch obj.waypoint_idx
        %         case 1
        %             obj.destination = [1000, 500, 0]; % First destination
        %         case 2
        %             obj.destination = obj.starting_pos; % Return to start
        %         case 3
        %             obj.destination = [500, 1000, 0]; % Second destination
        %         case 4
        %             obj.destination = obj.starting_pos; % Return to start
        %         case 5
        %             obj.destination = [500, 0, 0]; % Third destination
        %         case 6
        %             obj.destination = obj.starting_pos; % Return to start
        %         case 7
        %             obj.destination = [0, 500, 0]; % Fourth destination
        %         case 8
        %             obj.destination = obj.starting_pos; % Return to start
        %     end
            
        %     % Compute desired direction and angle
        %     vec_to_dest = obj.destination - obj.car_loc;
        %     dist_to_dest = norm(vec_to_dest);
            
        %     % Check if destination reached
        %     epsilon = 10.0; % Tolerance in meters
        %     if dist_to_dest < epsilon
        %         % We've reached the destination - move to next waypoint
        %         obj.waypoint_idx = mod(obj.waypoint_idx, 8) + 1;
        %         % disp(['*** WAYPOINT REACHED! Moving to waypoint #', num2str(obj.waypoint_idx), ' ***']);
        %     else
        %         % Continue moving toward destination
        %         % Normalize direction vector
        %         direction = vec_to_dest / dist_to_dest;
                
        %         % Calculate desired orientation
        %         desired_angle = atan2(direction(2), direction(1));
        %         current_angle = obj.car_orientation;
        %         angle_error = wrapToPi(desired_angle - current_angle);
                
        %         % PID control for steering
        %         Kp = 0.5; Ki = 0.01; Kd = 0.1;
        %         obj.integral_error = obj.integral_error + angle_error * obj.dt;
        %         derivative_error = (angle_error - obj.prev_error) / obj.dt;
        %         steering_angle = Kp * angle_error + Ki * obj.integral_error + Kd * derivative_error;
        %         steering_angle = max(-max_turning_angle, min(max_turning_angle, steering_angle));
        %         obj.car_orientation = obj.car_orientation + steering_angle;
        %         obj.prev_error = angle_error;
                
        %         % FORCE a non-zero speed (critical fix)
        %         obj.current_speed = 20; % Fixed speed for reliable movement
                
        %         % Calculate movement vector based on orientation and speed
        %         movement = [cos(obj.car_orientation), sin(obj.car_orientation), 0] * obj.current_speed * obj.dt;
                
        %         % Update car position
        %         obj.car_loc = obj.car_loc + movement;
        %     end
            
        %     obj.target_loc = obj.car_loc;
        %     obj.time = obj.time + obj.dt;
        %     obj.stepCount = obj.stepCount + 1;
            
        %     % Update trajectory and visualization
        %     obj.updateTrajectory();
        %     obj.updateVisualization();
            
        %     % Check if episode is done
        %     done = obj.stepCount >= obj.maxSteps;
            
        %     % Retrieve the next state
        %     next_state = getState(obj);
        % end  
        

        % function [next_state, reward, peb, rate, power, done] = step(obj, action)
        %     % Update RIS phases based on action
        %     ris_phases = action(1:obj.Nr);
        %     obj.phi = diag(exp(1j * 2 * pi * ris_phases));
        
        %     % Calculate performance metrics
        %     peb = obj.calculatePerformanceMetrics();
        %     rate = getrate(obj);
        %     power = getpower(obj);
        %     reward = obj.computeReward(peb);
        %     reward = abs(reward); % Ensure reward is a real-valued scalar
        
        %     % Vehicle dynamics parameters
        %     max_speed = 30; % Maximum speed in m/s
        %     max_acceleration = 2; % Maximum acceleration in m/s^2
        %     max_turning_angle = pi/6; % Maximum turning angle in radians
        
        %     % Compute desired direction and angle
        %     direction = (obj.destination - obj.car_loc) / norm(obj.destination - obj.car_loc);
        %     desired_angle = atan2(direction(2), direction(1));
        %     current_angle = obj.car_orientation;
        %     angle_error = wrapToPi(desired_angle - current_angle);
        
        %     % PID control for steering
        %     Kp = 0.5; Ki = 0.01; Kd = 0.1;
        %     obj.integral_error = obj.integral_error + angle_error * obj.dt;
        %     derivative_error = (angle_error - obj.prev_error) / obj.dt;
        %     steering_angle = Kp * angle_error + Ki * obj.integral_error + Kd * derivative_error;
        %     steering_angle = max(-max_turning_angle, min(max_turning_angle, steering_angle));
        %     obj.car_orientation = obj.car_orientation + steering_angle;
        %     obj.prev_error = angle_error;
        
        %     % Speed control
        %     target_speed = obj.speed;
        %     speed_diff = target_speed - obj.current_speed;
        %     acceleration = max(-max_acceleration, min(max_acceleration, speed_diff / obj.dt));
        %     obj.current_speed = max(0, min(max_speed, obj.current_speed + acceleration * obj.dt));
        
        %     % Update vehicle position
        %     obj.car_loc = obj.car_loc + [cos(obj.car_orientation), sin(obj.car_orientation), 0] * obj.current_speed * obj.dt;
        %     obj.target_loc = obj.car_loc;
        %     obj.time = obj.time + obj.dt;
        %     obj.stepCount = obj.stepCount + 1;
        
        %     % Update trajectory and visualization
        %     obj.updateTrajectory();
        %     obj.updateVisualization();
        
        %     % Check if destination reached or out of bounds
        %     epsilon = 10.0; % Tolerance in meters
        %     reached_dest = norm(obj.destination - obj.car_loc) < epsilon;
        %     x = obj.car_loc(1);
        %     y = obj.car_loc(2);
        %     out_of_bounds = (x < 0 || x > obj.env_dims(1) || y < 0 || y > obj.env_dims(2));
        
        %     if reached_dest || out_of_bounds
        %         % Assign a new random destination within environment bounds
        %         obj.destination = [rand() * obj.env_dims(1), rand() * obj.env_dims(2), 0];
        %     end

        %     done = obj.stepCount >= obj.maxSteps;
        
        %     % Retrieve the next state
        %     next_state = getState(obj);
        % end        

        function reward = computeReward(obj, peb)
            % Q = 0.5;
            constraints_satisfied = (obj.rate >= obj.R_min);

            base_reward = 1 / peb;  % Keeps reward bounded [0,1]
            n = rand(1);
            
            if ~constraints_satisfied
                % reward = base_reward * (0.5 + 0.5 * (obj.rate / obj.R_min)); 
                reward = base_reward * n;
            else
                reward = base_reward;
            end

        end
        
        function state = reset(obj)
            % Reset simulation state
            obj.initializeChannels();
            obj.initializephi();
            obj.calculate_pathloss();
            % obj.destination = [randi([0, 1000]), randi([0, 1000]), 0];
            % obj.initializeVisualization();
            % obj.destination = [999, 999, 0];
            obj.H_combined = obj.compute_Heff();
            [obj.Wx,obj.W] = computeWx(obj);
            obj.Pb = obj.getpower();
            obj.gamma_c = obj.computeSNR();
            obj.rate = obj.getrate();
            obj.peb = obj.calculatePerformanceMetrics(obj.Wx);
            % obj.calculated_values();
            % obj.destination = [randi([0, 1000]), randi([0, 1000]), 0];

            obj.car_loc = obj.starting_pos;
            obj.target_loc = obj.car_loc;
            obj.time = 0;
            obj.stepCount = 0;

            
            % Return the current state
            state = obj.getState();
        end
        % ! -------------------- MACHINE LEARNING PART ENDS HERE --------------------

        % ! -------------------- PEB COMPUTATION PART STARTS HERE --------------------
        

        function [R_min] = computeR_min(obj)
            % Shannon capacity formula for baseline
            R_theoretical = obj.B * log2(1 + obj.gamma_c);

            % Set R_min as a fraction of theoretical maximum
            R_min = 0.5 * R_theoretical;
        end


        function [L1, L2, L3, L_proj1, L_proj2, L_proj3, delays, angles] = computeGeometricParameters(obj)
            % Extract coordinates
            xb = obj.bs_loc(1);    yb = obj.bs_loc(2);    zb = obj.bs_loc(3);
            xr = obj.ris_loc(1);   yr = obj.ris_loc(2);   zr = obj.ris_loc(3);
            xt = obj.target_loc(1); yt = obj.target_loc(2); zt = obj.target_loc(3);

            % disp(xt)
            % disp(yt)
            
            % Calculate 3D Euclidean distances
            L1 = sqrt((xb - xr)^2 + (yb - yr)^2 + (zb - zr)^2);
            L2 = sqrt((xr - xt)^2 + (yr - yt)^2 + zr^2);
            L3 = sqrt((xb - xt)^2 + (yb - yt)^2 + zb^2);
            
            % Calculate 2D (X-Y plane) projections
            L_proj1 = sqrt((xb - xr)^2 + (yb - yr)^2);
            L_proj2 = sqrt((xr - xt)^2 + (yr - yt)^2);
            L_proj3 = sqrt((xb - xt)^2 + (yb - yt)^2);
            
            delays.line_of_sight = L3 / obj.c;
            delays.non_line_of_sight = (L1 + L2) / obj.c;
            
            angles.bs_to_ris.azimuth = asin((zb - zr) / L1);
            angles.bs_to_ris.elevation_azimuth = asin((xb - xr) / L_proj1);
            angles.bs_to_ris.elevation_angle = acos((zb - zr) / L1);
            
            angles.ris_to_target.aoa = asin(zr / L2);
            angles.ris_to_target.azimuth = acos((yr - yt) / L_proj2);
            angles.ris_to_target.elevation_angle = acos(zr / L2);

            angles.bs_to_target_transmit = acos(zb/L3);
            angles.bs_to_target_receive = asin(zb/L3);
        end

        function [L1, L2, L3, L1_t, L2_t, L_proj1, L_proj2, L_proj3, delays, angles] = computeGeometricParametersStar(obj)
            % Extract coordinates
            xb = obj.bs_loc(1);    yb = obj.bs_loc(2);    zb = obj.bs_loc(3);
            xr = obj.ris_loc(1);   yr = obj.ris_loc(2);   zr = obj.ris_loc(3);
            xt = obj.target_loc(1); yt = obj.target_loc(2); zt = obj.target_loc(3);
            
            % Reflection path
            L1 = sqrt((xb - xr)^2 + (yb - yr)^2 + (zb - zr)^2);
            L2 = sqrt((xr - xt)^2 + (yr - yt)^2 + zr^2);
            L3 = sqrt((xb - xt)^2 + (yb - yt)^2 + zb^2);
        
            % Transmission path  
            L1_t = sqrt((xb - xr)^2 + (yb - yr)^2 + (zb - zr)^2);
            L2_t = sqrt((xr - xt)^2 + (yr - yt)^2 + (zr - zt)^2);  % Different due to transmission

            % Calculate 2D (X-Y plane) projections
            L_proj1 = sqrt((xb - xr)^2 + (yb - yr)^2);
            L_proj2 = sqrt((xr - xt)^2 + (yr - yt)^2);
            L_proj3 = sqrt((xb - xt)^2 + (yb - yt)^2);

            L_r = sqrt((xb - xr)^2 + (yb - yr)^2 + (zb - zr)^2) + sqrt((xr - xt)^2 + (yr - yt)^2 + zr^2);
            L_t = sqrt((xb - xr)^2 + (yb - yr)^2 + (zb - zr)^2) + sqrt((xr - xt)^2 + (yr - yt)^2 + (zr - zt)^2);

            delays.reflected = L_r / obj.c;
            delays.transmitted = L_t / obj.c;

            % Calculate delays
            delays.line_of_sight = L3 / obj.c;
            delays.non_line_of_sight = (L1 + L2) / obj.c;
            delays.transmitted = (L1_t + L2_t) / obj.c;
        
            % Angles (Reflection)
            angles.bs_to_ris.azimuth = asin((zb - zr) / L1);
            angles.bs_to_ris.elevation_azimuth = asin((xb - xr) / L_proj1);
            angles.bs_to_ris.elevation_angle = acos((zb - zr) / L1);
            
            angles.ris_to_target.aoa = asin(zr / L2);
            angles.ris_to_target.azimuth = acos((yr - yt) / L_proj2);
            angles.ris_to_target.elevation_angle = acos(zr / L2);
        
            % Angles (Transmission)
            angles.bs_to_star_ris = angles.bs_to_ris;  % Same as reflection
            angles.star_ris_to_target.aoa = asin((zr - zt) / L2_t);
            angles.star_ris_to_target.elevation_angle = acos((zr - zt) / L2_t);
            angles.star_ris_to_target.transmission_elevation = acos((zr - zt) / L2_t);
        end
        

        function [peb] = calculatePerformanceMetrics(obj, Wx)
            [J, ~, ~] = computeFisherInformationMatrix(obj,Wx);
            CRLB = inv(J);
            obj.peb = sqrt(trace(CRLB));
            obj.peb = real(obj.peb);
            % rate_constraint_satisfied = (obj.rate >= obj.R_min);

            % if ~rate_constraint_satisfied
            %     penalty_factor = 1 + (obj.R_min - obj.rate)/obj.R_min;
            %     obj.peb = obj.peb * penalty_factor;
            % end
            % obj.peb = sqrt((real(obj.peb))^2 + (imag(obj.peb))^2)*100;
            % obj.peb = 1+11/(1+exp(-(obj.peb-6.5)));
            % if(obj.peb < obj.minpeb)
            %     obj.minpeb = obj.peb;
            % else
            %     obj.peb = obj.minpeb;
            % end
            peb = obj.peb;
            % disp(peb);
        end
        
        function [T] = computeTransformationMatrix(obj)
            T = zeros(2, 7);
            xb = obj.bs_loc(1);    yb = obj.bs_loc(2);    zb = obj.bs_loc(3);
            xr = obj.ris_loc(1);   yr = obj.ris_loc(2);   zr = obj.ris_loc(3);
            xt = obj.target_loc(1); yt = obj.target_loc(2);
            % disp(xt)
            % disp(yt)

            [~, L2, L3, ~, L_proj2, ~, ~, ~] = computeGeometricParameters(obj);

            T(1,1) = (xt-xb) / (obj.c*L3);
            T(1,2) = (xt-xr) / (obj.c*L2);
            T(1,3) = (zr*(xt-xr)) / ((L2^3)*sqrt(1-((zr)^2)/(L2)^2));
            T(1,4) = ((yr-yt)*(xr-xt)) / ((L2^3)*sqrt(1-((yr-yt)^2)/(L2)^2));
            T(1,5) = (zr*(xr-xt)) / ((L2^3)*sqrt(1-((zr)^2)/(L2)^2));
            T(1,6) = (zb*(xt-xb)) / ((L3^3)*sqrt(1-L3^2));
            T(1,7) = (zb*(xt-xb)) / ((L3^3)*sqrt(1-L3^2));
            
            T(2,1) = (yt-yb) / (obj.c*L3);
            T(2,2) = (yt-yr) / (obj.c*L2);
            T(2,3) = (zr*(yt-yr)) / ((L2^3)*sqrt(1-((zr)^2)/(L2)^2));
            T(2,4) = ((L_proj2^2) * (yr-yt)*(xr-xt)) / ((L2^3)*sqrt(1-((yr-yt)^2)/(L2)^2));
            T(2,5) = (zr*(yr-yt)) / ((L2^3)*sqrt(1-((zr)^2)/(L2)^2));
            T(2,6) = (zb*(yt-yb)) / ((L3^3)*sqrt(1-L3^2));
            T(2,7) = (zb*(yt-yb)) / ((L3^3)*sqrt(1-L3^2));

            % disp ("this is L2")
            % disp(L2)
            % disp("this is L3")
            % disp(L3)
            % disp("this is Lproj2")
            % disp(L_proj2)
            % disp(T)

        end
        
        % ! Don't remove this
        function [Wx, W] = computeWx(obj)
            N = obj.Ns;
            if obj.stepCount > 1
                W_mrt = compute_Heff(obj);
                W_mrt = W_mrt / norm(W_mrt, 'fro');
                W_mrt_reshaped = W_mrt(:,1);
                W = repmat(W_mrt_reshaped, [1, obj.Ns]);
                % disp("This is the size after first step")
                % disp(size(W));
            else
                W = rand(obj.Nb, obj.Ns) + 1j*randn(obj.Nb, obj.Ns);
                % disp("This is the size at first step")
                % disp(size(W));
            end  
            W = W ./ vecnorm(W); 
            Wx = zeros(obj.Nb, obj.Ns);
            % Generate a different X for each subcarrier
            for n = 1:N
                X_n = (randn(obj.Ns, 1) + 1j*randn(obj.Ns, 1)) / sqrt(2);
                Wx(:,n) = W * X_n;
            end
        end
        
        % ! Don't remove this either
        % function [Wx, W] = computeWx(obj)
        %     N = obj.Ns;  % Number of subcarriers
            
        %     % Define subarray structure
        %     numSubarrays = 2;  % Number of subarrays (RF chains)
        %     elementsPerSubarray = obj.Nb / numSubarrays;  % Elements per subarray
            
        %     % 1. Digital precoding matrix (baseband processing)
        %     W_BB = rand(numSubarrays, obj.Mb) + 1j*randn(numSubarrays, obj.Mb);
        %     W_BB = W_BB ./ vecnorm(W_BB);  % Normalize digital weights
            
        %     % 2. Analog beamforming matrix (RF domain with phase shifters)
        %     W_RF = zeros(obj.Nb, numSubarrays);
            
        %     % Create analog beamforming matrix with phase-only constraints
        %     for i = 1:numSubarrays
        %         % Calculate which elements belong to this subarray
        %         startIdx = (i-1)*elementsPerSubarray + 1;
        %         endIdx = i*elementsPerSubarray;
                
        %         % Generate random phases (analog phase shifters can only change phase)
        %         phases = 2*pi*rand(elementsPerSubarray, 1);
                
        %         % Set the phase shifters for this subarray (unit magnitude)
        %         W_RF(startIdx:endIdx, i) = exp(1j*phases);
        %     end
            
        %     % Combined hybrid beamforming matrix
        %     W = W_RF * W_BB;
            
        %     % Initialize Wx with proper dimensions
        %     Wx = zeros(obj.Nb, N);
            
        %     % Generate a different X for each subcarrier
        %     for n = 1:N
        %         X_n = (randn(obj.Mb, 1) + 1j*randn(obj.Mb, 1)) / sqrt(2);
                
        %         % Apply hybrid beamforming
        %         Wx(:,n) = W * X_n;
        %     end
        % end        
        
        function [J, J_zao, T] = computeFisherInformationMatrix(obj,Wx)
            % sigma_s = sqrt(obj.Pb/obj.gamma_c);  % Noise variance (placeholder)
            [T] = computeTransformationMatrix(obj);
            gamma_l = sqrt(obj.Nb*obj.Nt)/sqrt(obj.rho_l);
            gamma_nl = sqrt(obj.Nb*obj.Nt)/sqrt(obj.rho_nl);
            
            [A1, A2, A3, A4] = computeAmplitudeMatrices(obj, obj.Ns, obj.B, gamma_l, gamma_nl, obj.h_l, obj.h_nl);
            [J_zao] = calculateJacobianMatrix(obj, obj.Pb, obj.Ns, Wx, A1, A2, A3, A4);
            % H_Los_3d = generate_H_Los(obj, obj.H_bt, obj.Nt, obj.Nb);
            % H_NLoS_3d = generate_H_NLoS(obj, obj.H_rt, obj.H_br, obj.Nt, obj.Nr, obj.Nb);
            % J_zao = computeJZao(obj, H_Los_3d, H_NLoS_3d);

            % Compute final Fisher Information Matrix
            J = T * J_zao * T';
        end

        % function J_zao = computeJZao(obj, H_Los, H_NLoS)
        %     % Extract parameters from object
        %     B = obj.B;
        %     fc = obj.fc;
        %     c = physconst('LightSpeed');
        %     N_ant_ris = 64;  % RIS elements
            
        %     % Calculate frequency-dependent SNRs
        %     snr_direct = squeeze(mean(abs(H_Los).^2, [1 2])) / obj.noise_var;  % [Ns×1]
        %     snr_reflected = squeeze(mean(abs(H_NLoS).^2, [1 2])) / obj.noise_var;
            
        %     % Average SNR across subcarriers
        %     avg_snr_direct = mean(snr_direct);
        %     avg_snr_reflected = mean(snr_reflected);
            
        %     % Calculate parameter variances (CRLB)
        %     lambda = c/fc;
        %     var_delay_direct = 1./(8*pi^2 * B^2 * snr_direct);
        %     var_delay_reflected = 1./(8*pi^2 * B^2 * snr_reflected);
            
        %     var_angle_direct = lambda^2./(8*pi^2 * snr_direct * (lambda/2)^2);
        %     var_angle_reflected = lambda^2./(8*pi^2 * snr_reflected * ((lambda/2)^2 * N_ant_ris));
            
        %     % Frequency-average variances
        %     sigma_sq = [
        %         mean(var_delay_direct);     % BS-target delay
        %         mean(var_delay_reflected);  % RIS-target delay
        %         mean(var_angle_reflected);  % RIS angle 1
        %         mean(var_angle_reflected);  % RIS angle 2
        %         mean(var_angle_reflected);  % RIS angle 3
        %         mean(var_angle_direct);     % BS elevation 1
        %         mean(var_angle_direct)      % BS elevation 2
        %     ];
            
        %     J_zao = diag(1./sigma_sq);
        % end
           
        
        function [A1, A2, A3, A4] = computeAmplitudeMatrices(obj, N, B, gamma_l, gamma_nl, h_l, h_nl)
            [~, ~, ~, ~, ~, ~, delays, ~] = computeGeometricParameters(obj);
            tau_l = delays.line_of_sight;
            tau_nl = delays.non_line_of_sight;
            A1 = zeros(1, N);
            A2 = zeros(1, N);
            A3 = zeros(1, N);
            A4 = zeros(1, N);
            for n = 1:N
                % Frequency-dependent scaling factor
                freq_factor = (n / N);
                
                % Compute each amplitude matrix entry
                A1(n) = gamma_l * h_l * (1j * 2 * pi * B * freq_factor) * ...
                        exp(1j * 2 * pi * B * freq_factor * tau_l);
                A2(n) = gamma_nl * h_nl * (1j * 2 * pi * B * freq_factor) * ...
                        exp(1j * 2 * pi * B * freq_factor * tau_nl);
                A3(n) = gamma_nl * h_nl * ...
                        exp(1j * 2 * pi * B * freq_factor * tau_nl);
                A4(n) = gamma_l * h_l * ...
                        exp(1j * 2 * pi * B * freq_factor * tau_l);
            end
        end 

        function [J_zao] = calculateJacobianMatrix(obj, Pb, N, Wx, A1, A2, A3, A4)
            % TODO: Initialize 7x7 Jacobian matrix
            J_zao = zeros(7, 7);
            [~, ~, ~, ~, ~, ~, ~, angles] = computeGeometricParameters(obj);
            psi_rt = angles.ris_to_target.aoa;
            psi_bt = angles.bs_to_target_transmit;
            psi_tb = angles.bs_to_target_receive;
            psi_br = angles.bs_to_ris.azimuth;
            phi_rt_a = angles.ris_to_target.azimuth;
            phi_rt_e = angles.ris_to_target.elevation_angle;
            phi_br_a = angles.bs_to_ris.elevation_azimuth;
            phi_br_e = angles.bs_to_ris.elevation_angle;

            % Initialize a_rt with proper dimensions
            % Initialize a_rt, a_bt, and a_tb with proper dimensions
            a_rt = zeros(obj.Nt, obj.Ns);
            a_bt = zeros(obj.Nb, obj.Ns);
            a_tb = zeros(obj.Nt, obj.Ns);

            % Calculate for each subcarrier
            for n = 1:obj.Ns
                % For a_rt - we need a column vector, not a diagonal matrix
                indices = (0:(obj.Nt-1))';  % Column vector of indices
                a_rt(:,n) = 1j * (2 * pi / obj.lambda) * cos(psi_rt) * indices;
                
                % For a_bt
                indices_b = (0:(obj.Nb-1))';
                a_bt(:,n) = 1j * (2 * pi / obj.lambda) * cos(psi_bt) * indices_b;
                
                % For a_tb
                a_tb(:,n) = 1j * (2 * pi / obj.lambda) * cos(psi_tb) * indices;
            end

            % Initialize arrays for a_rt_a and a_rt_e
            a_rt_a = zeros(obj.Nr, obj.Ns);
            a_rt_e = zeros(obj.Nr, obj.Ns);

            % Calculate a_rt_a and a_rt_e for each subcarrier
            for n = 1:obj.Ns
                a_rt_a(:, n) = 1j * (2 * pi / obj.lambda) * obj.lambda/2 * ((obj.Nx-1) * cos(phi_rt_a) * sin(phi_rt_e));
                a_rt_e(:, n) = 1j * (2 * pi / obj.lambda) * obj.lambda/2 * (((obj.Nx-1) * sin(phi_rt_a) * cos(phi_rt_e)) - ((obj.Ny-1) * sin(phi_rt_e)));
            end

            % TODO: implement all the a vector
            a_vec = compute_a_psi(obj, obj.Nt, psi_bt, obj.lambda, obj.lambda/2);
            a_psi_bt = a_vec;
            a_vec = compute_a_psi(obj, obj.Nt, psi_tb, obj.lambda, obj.lambda/2);
            a_psi_tb = a_vec;
            a_vec = compute_a_psi(obj, obj.Nt, psi_rt, obj.lambda, obj.lambda/2);
            a_psi_rt = a_vec;
            a_vec = compute_a_psi(obj, obj.Nb, psi_br, obj.lambda, obj.lambda/2);
            a_psi_br = a_vec;


            % TODO: implement all the steering vector
            a_phi_br = compute_a_phi(obj, obj.Nx, phi_br_a, phi_br_e, obj.lambda, obj.lambda/2);
            a_phi_rt = compute_a_phi(obj, obj.Nx, phi_rt_a, phi_rt_e, obj.lambda, obj.lambda/2);

            for n = 1:N
                % Calculate all partial derivatives
                d_mu_array = cell(7, 1);
                
                % Extract the nth column from each steering vector matrix
                a_psi_bt_n = a_psi_bt(:,n);
                a_psi_tb_n = a_psi_tb(:,n);
                a_psi_rt_n = a_psi_rt(:,n);
                a_psi_br_n = a_psi_br(:,n);
                a_phi_rt_n = a_phi_rt(:,n);
                a_phi_br_n = a_phi_br(:,n);
                
                % Extract the nth element from amplitude matrices
                A1_n = A1(n);
                A2_n = A2(n);
                A3_n = A3(n);
                A4_n = A4(n);
                
                % Partial derivatives with respect to each parameter
                % d_mu_d_tau_l
                d_mu_array{1} = (A1_n * a_psi_bt_n) * (a_psi_tb_n' * Wx(:,n));
                
                % d_mu_d_tau_nl
                d_mu_array{2} = (A2_n * a_psi_rt_n) * (a_phi_rt_n' * obj.phi * a_phi_br_n) * (a_psi_br_n' * Wx(:,n));
                
                d_mu_array{3} = A3_n * (a_rt(:,n) .* a_psi_rt_n) * (a_phi_rt_n' * obj.phi * a_phi_br_n) * (a_psi_br_n' * Wx(:,n));

                % d_mu_d_phi_rt_a
                d_mu_array{4} = (A3_n * a_psi_rt_n) * (a_phi_rt_n' * diag(a_rt_a(:,n)) * obj.phi * a_phi_br_n) * (a_psi_br_n' * Wx(:,n));
                
                % d_mu_d_phi_rt_e
                d_mu_array{5} = (A3_n * a_psi_rt_n) * (a_phi_rt_n' * diag(a_rt_e(:,n)) * obj.phi * a_phi_br_n) * (a_psi_br_n' * Wx(:,n));
                
                % d_mu_d_psi_br
                d_mu_array{6} = (A4_n * a_bt(:,n)' * a_psi_bt_n) * (a_psi_tb_n' * Wx(:,n));
                
                d_mu_array{7} = A4_n * (a_tb(:,n) .* a_psi_bt_n) * (a_psi_tb_n' * Wx(:,n));

                % Calculate Jacobian matrix elements
                for i = 1:7
                    for j = 1:7
                        % Get the sizes of the current derivatives
                        [rows_i, cols_i] = size(d_mu_array{i});
                        [rows_j, cols_j] = size(d_mu_array{j});
                        
                        % Compute the FIM entry based on dimensions
                        if rows_i == rows_j && cols_i == cols_j
                            % If dimensions match, use dot product
                            J_zao(i,j) = J_zao(i,j) +  (2*obj.Pb/obj.sigma_c_sq)*real(sum(sum(conj(d_mu_array{i}) .* d_mu_array{j})));
                        else
                            J_zao(i,j) = J_zao(i,j) +  (2*obj.Pb/obj.sigma_c_sq)*real(sum(sum(d_mu_array{i} * d_mu_array{j}')));
                        end
                    end
                end                
            end
            % obj.Jzao = J_zao;
            % disp("this is Jzao matrix: ");
            % disp(J_zao);
        end                         
        % ! -------------------- PEB COMPUTATION PART ENDS HERE --------------------        

    end
    
end
