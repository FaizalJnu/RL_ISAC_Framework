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
        sigma_c_sq_linear = 10e-6;
        sigma_c_sq = -60; 
        Pb_dbm = 0;

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
        J
        R_min = 0
        H_combined
        J_zao
        waypoint_idx = 1;
        state_mean
        state_std
        peb_zero = 0

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
        episodeCount = 0;

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
            obj.initializeVisualization();
            obj.initializephi();
            obj.calculate_pathloss();
            % obj.destination = [randi([0, 1000]), randi([0, 1000]), 0];
            % obj.destination = [999, 999, 0];
            [H_Los,H_Los_3d] = generate_H_Los(obj, obj.H_bt, obj.Nt, obj.Nb);
            [H_NLos,H_NLos_3d] = generate_H_NLoS(obj, obj.H_rt, obj.H_br, obj.Nt, obj.Nr, obj.Nb, obj.phi);
            obj.H_combined = obj.compute_Heff(H_Los, H_NLos);
            [obj.Wx,obj.W] = computeWx(obj);
            obj.Pb = obj.getpower();
            obj.gamma_c = computeSNR(obj);
            obj.rate = obj.getrate();
            obj.peb = obj.calculatePerformanceMetrics(obj.Wx, H_Los_3d, H_NLos_3d);
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

        function H_combined = compute_Heff(obj, H_Los, H_NLos)
            H_combined = H_Los + H_NLos;
            obj.H_combined = H_combined;
        end

        function gamma_c = computeSNR(obj)
            [~,HLos_3d] = generate_H_Los(obj, obj.H_bt, obj.Nt, obj.Nb);
            [~,HNLos_3d] = generate_H_NLoS(obj, obj.H_rt, obj.H_br, obj.Nt, obj.Nr, obj.Nb, obj.phi);
            obj.Pb = getpower(obj);
            gamma_c_per_subcarrier = zeros(1, obj.Ns);
            for n = 1:obj.Ns
                H_combined_n = HLos_3d(:,:,n) + HNLos_3d(:,:,n);
                gamma_c_per_subcarrier(n) = obj.Pb * norm(H_combined_n * obj.W, 'fro')^2 / obj.sigma_c_sq_linear;
            end
            obj.gamma_c = obj.Ns / sum(1 ./ gamma_c_per_subcarrier);  
            obj.SNR = 10 * log10(obj.gamma_c);
            gamma_c = obj.gamma_c;
        end

        function Pb = getpower(obj)
            % Pb = trace(obj.Wx * obj.Wx') / obj.Ns;
            obj.Pb = 1;
            obj.Pb_dbm = 30;
            Pb = obj.Pb;

        end

        function rate = getrate(obj)
            rate = obj.B*log2(1+obj.gamma_c);
            obj.rate = real(rate);
            % disp(rate);
        end

        % ! -------------------- CHANNEL INITIALIZATION PART STARTS HERE --------------------        
        function initializeChannels(obj)
            [obj.H_bt, obj.H_br, obj.H_rt] = generate_channels(obj, obj.Nt, obj.Nr, obj.Nb);               
        end

        function initializephi(obj)
            rho_r = 1;
            Bit = 2;  
            Delta_delta = 2*pi / (2^Bit);  
            A = (0 : (2^Bit - 1)) * Delta_delta;
            theta = A(randi(numel(A), obj.Nr, 1));
            u = rho_r * exp(1j*theta);
            obj.phi = diag(u); 
            % disp(obj.phi);
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
        
            % Frequency-independent steering vectors (same for all subcarriers)
            a_psi_bt = compute_a_psi(obj, Nb, psi_bt, lambda, d);  % (Nb × Ns)
            a_psi_tb = compute_a_psi(obj, Nt, psi_tb, lambda, d);  % (Nt × Ns)
        
            % Outer product (identical across subcarriers)
            H_single = a_psi_tb(:,1) * a_psi_bt(:,1)';  % Nt × Nb
        
            % Replicate for all subcarriers
            H_bt = repmat(H_single, 1, 1, obj.Ns);      % Nt × Nb × Ns
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
            % Frequency-independent steering vectors
            a_psi_br = compute_a_psi(obj, Nb, psi_br, lambda, d);                    % Nb × Ns
            a_phi_abr = compute_a_phi(obj, sqrt(Nr), phi_abr, phi_ebr, lambda, dr); % Nr × Ns
        
            % Outer product (same for all subcarriers)
            H_single = a_phi_abr(:,1) * a_psi_br(:,1)';  % Nr × Nb
        
            % Replicate for all subcarriers
            H_br = repmat(H_single, 1, 1, obj.Ns);       % Nr × Nb × Ns
        end 
        
        function H_rt = generate_H_rt(obj, Nt, Nr, phi_art, phi_ert, psi_rt, lambda, dr, d)
            % Frequency-independent steering vectors
            a_psi_rt = compute_a_psi(obj, Nt, psi_rt, lambda, d);                    % Nt × Ns
            a_phi_art = compute_a_phi(obj, sqrt(Nr), phi_art, phi_ert, lambda, dr); % Nr × Ns
        
            % Outer product (same for all subcarriers)
            H_single = a_psi_rt(:,1) * a_phi_art(:,1)';  % Nt × Nr
        
            % Replicate for all subcarriers
            H_rt = repmat(H_single, 1, 1, obj.Ns);       % Nt × Nr × Ns
        end
        

        function a_vec = compute_a_psi(obj, Nant, psi, ~, d)
            k = 2*pi*obj.fc / 3e8;         % Wavenumber at center frequency
            n_ant = (0:(Nant-1)).';        % Antenna indices
            a = exp(1j * k * d * n_ant * sin(psi)) / sqrt(Nant);
            a_vec = repmat(a, 1, obj.Ns);  % Same steering vector across all subcarriers
        end
        

        function a_phi = compute_a_phi(obj, Nx, phi_a, phi_e, lambda, dr)
            fc = obj.fc;
            N2 = Nx * Nx;
            Ns = obj.Ns;
        
            k = 2 * pi * fc / 3e8;  % Wavenumber at center frequency
        
            a_phi_single = zeros(N2, 1);  % Steering vector at fc
            idx = 1;
            for m_idx = 1:Nx
                for n_idx = 1:Nx
                    phase_term = exp(1j * k * dr * ...
                        (m_idx * sin(phi_a) * sin(phi_e) + n_idx * cos(phi_e)));
                    a_phi_single(idx) = phase_term;
                    idx = idx + 1;
                end
            end
        
            a_phi_single = a_phi_single / sqrt(N2);         % Normalize
            a_phi = repmat(a_phi_single, 1, Ns);            % Copy across all subcarriers
        end             
        
        function [H_Los, H_Los_3d] = generate_H_Los(obj, H_bt, Nt, Nb)
            gamma_l = sqrt(Nb * Nt) / sqrt(obj.rho_l);
        
            [~,~,~,~, ~, ~, delays, ~] = computeGeometricParameters(obj);
            tau_l = delays.line_of_sight;
        
            H_Los_3d = zeros(Nt, Nb, obj.Ns);
        
            for n = 1:obj.Ns
                phase = exp(1j * 2 * pi * obj.B * (n / obj.Ns) * tau_l);
                H_Los_3d(:,:,n) = gamma_l * obj.h_l * H_bt(:,:,n) * phase;
            end
        
            H_Los = mean(H_Los_3d, 3);  % Averaged channel
        end
        
        
        function [H_NLoS, H_NLoS_3d] = generate_H_NLoS(obj, H_rt, H_br, Nt, Nr, Nb, phi)
            obj.rho_nl = 4;  % Set if not already set externally
        
            gamma_nl = sqrt(Nb * Nr) / sqrt(obj.rho_nl);
        
            [~, ~, ~, ~, ~, ~, delays, ~] = computeGeometricParameters(obj);
            tau_nl = delays.non_line_of_sight;
        
            H_NLoS_3d = zeros(Nt, Nb, obj.Ns);
        
            for n = 1:obj.Ns
                % Frequency-dependent phase shift
                phase = exp(1j * 2 * pi * obj.B * (n / obj.Ns) * tau_nl);
                
                % NLoS channel: Tx → RIS → Rx
                H_NLoS_3d(:,:,n) = gamma_nl * obj.h_nl * H_rt(:,:,n) * phi * H_br(:,:,n) * phase;
            end
        
            % Mean over subcarriers (narrowband equivalent)
            H_NLoS = mean(H_NLoS_3d, 3);
        end
        
               
        

        % ! -------------------- CHANNEL INITIALIZATION PART ENDS HERE --------------------        
        
        % ! -------------------- MACHINE LEARNING PART STARTS HERE --------------------        
        function state = getState(obj)
            phi_real = real(diag(obj.phi))';
            phi_min = min(phi_real(:));
            phi_max = max(phi_real(:));
            phi_real = (phi_real-phi_min) / (phi_max-phi_min);
            phi_real = phi_real*2 - 1;

            phi_imag = imag(diag(obj.phi))';
            phi_min = min(phi_imag(:));
            phi_max = max(phi_imag(:));
            phi_imag = (phi_imag-phi_min) / (phi_max-phi_min);
            phi_imag = phi_imag*2 - 1;
            
            H = obj.H_combined;
            
            H_real = real(H(:))';
            H_min = min(H_real(:));
            H_max = max(H_real(:));
            H_real = (H_real-H_min) / (H_max-H_min);
            H_real = H_real*2 - 1;

            H_imag = imag(H(:))';
            H_min = min(H_imag(:));
            H_max = max(H_imag(:));
            H_imag = (H_imag-H_min) / (H_max-H_min);
            H_imag = H_imag*2 - 1;

            Rc = obj.rate;
            Rc_norm = Rc / 1e8;
            Rc_norm = Rc_norm / 5;
            Rc_norm = Rc_norm*2 - 1;
            % disp(obj.phi);
            
            state = [phi_real, ...
                    phi_imag, ...
                    Rc_norm, ...
                    H_real, ...
                     H_imag];

            state = real(state); 
            state = state(:)';
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

        function pzc = getpebzero(obj)
            pzc = obj.peb_zero;
        end

        function log_step_metrics(obj, action, reward, peb, rate, power)
            % Create filename based on current episode number
            filename = sprintf('Episode_%04d_log.txt', obj.episodeCount); 
            
            % Open file in append mode
            fid = fopen(filename, 'a');
        
            % Write current step number
            fprintf(fid, 'Step %d:\n', obj.stepCount);
        
            % Write metrics in readable form
            fprintf(fid, 'Reward: %.6f\n', reward);
            fprintf(fid, 'PEB: %.6f\n', peb);
            fprintf(fid, 'Rate: %.6f bps/Hz\n', rate);
            fprintf(fid, 'Power: %.6f dBm\n', power);
        
            % Write action vector
            fprintf(fid, 'Action: [');
            fprintf(fid, '%.4f ', action);
            fprintf(fid, ']\n');

            fprintf(fid, 'Jzao: [');
            fprintf(fid, '%.4f ', obj.J_zao);
            fprintf(fid, ']\n');

            fprintf(fid, 'H_combined: [');
            fprintf(fid, '%.4f ', obj.H_combined);
            fprintf(fid, ']\n');
            
            fprintf(fid, 'Fisher Information Matrix: [');
            fprintf(fid, '%.4f ', obj.J);
            fprintf(fid, ']\n');
        
            % Add separator
            fprintf(fid, '--------------------------------------------\n\n');
        
            fclose(fid);
        end        
        
        function [next_state, reward, peb, rate, power, done] = step(obj, action)
            % == Phase calculation ==
            ris_phases = action(1:64); % values in [-1, 1]
            normalized_phases = (ris_phases + 1) / 2; % now in [0, 1]
            u = exp(1j*2*pi*normalized_phases);
            obj.phi = diag(u);          
        
            % == Channel generation ==
            [H_Los, H_Los_3d] = generate_H_Los(obj, obj.H_bt, obj.Nt, obj.Nb);
            [H_NLos, H_NLos_3d] = generate_H_NLoS(obj, obj.H_rt, obj.H_br, obj.Nt, obj.Nr, obj.Nb, obj.phi);
            obj.H_combined = compute_Heff(obj, H_Los, H_NLos);
            [obj.Wx, obj.W] = computeWx(obj);
        
            % == Power & metrics ==
            power = getpower(obj);
            obj.Pb = power;
            power = obj.Pb_dbm;
            obj.gamma_c = computeSNR(obj);
            rate = getrate(obj);
            obj.rate = rate;
        
            % == PEB and Reward ==
            peb = obj.calculatePerformanceMetrics(obj.Wx, H_Los_3d, H_NLos_3d);
            obj.peb = peb;
            if peb == 0
                obj.peb_zero = obj.peb_zero + 1;
            end
            reward = abs(obj.computeReward(peb));
        
            % == Logging ==
            log_step_metrics(obj, action, reward, peb, rate, power);
            % Initialize position once per episode
            persistent isInitialized;
            if isempty(isInitialized) || obj.stepCount == 0
                isInitialized = true;
                obj.car_loc = [500, 500, 0]; % Start position
                obj.car_orientation = 0;     % Facing along positive x-axis
                obj.stepCount = 0;
            end
        
            % Move car linearly along x-axis by 0.05 meters per step
            obj.car_loc(1) = obj.car_loc(1) + 0.05;
        
            % Keep y and z fixed (assuming flat ground at y = 500, z = 0)
            obj.car_loc(2) = 500;
            obj.car_loc(3) = 0;
        
            % Update orientation (still facing x-direction)
            obj.car_orientation = 0;
        
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
        %     [H_Los, H_Los_3d] = generate_H_Los(obj, obj.H_bt, obj.Nt, obj.Nb);
        %     [H_NLos, H_NLos_3d] = generate_H_NLoS(obj, obj.H_rt, obj.H_br, obj.Nt, obj.Nr, obj.Nb, obj.phi);
        %     obj.H_combined = compute_Heff(obj, H_Los, H_NLos);
        %     [obj.Wx,obj.W] = computeWx(obj);
        %     power = getpower(obj);
        %     obj.Pb = power;
        %     power = obj.Pb_dbm;
        %     obj.gamma_c = computeSNR(obj);
        %     rate = getrate(obj);
        %     obj.rate = rate;
        %     peb = obj.calculatePerformanceMetrics(obj.Wx, H_Los_3d, H_NLos_3d);
        %     obj.peb = peb;
        %     % Calculate performance metrics
        %     reward = obj.computeReward(peb);
        %     reward = abs(reward); % Ensure reward is a real-valued scalar
            
        %     % Initialize properties only once per episode
        %     persistent isInitialized;
        %     if isempty(isInitialized) || obj.stepCount == 0
        %         isInitialized = true;
        %         obj.center = [250, 500, 0]; % Center of circular path
        %         obj.radius = 250; % Radius of circular path in meters
        %         obj.angular_speed = 0.08; % Angular speed in radians per second
        %         obj.angle = 0; % Initial angle
        %         % Force car to start at a position on the circle
        %         obj.car_loc = obj.center + [obj.radius, 0, 0];
        %         obj.car_orientation = pi/2; % Start with orientation tangent to circle
        %     end
            
        %     % Update angle for circular motion
        %     obj.angle = obj.angle + obj.angular_speed * obj.dt;
            
        %     % Calculate new position on the circle
        %     new_x = obj.center(1) + obj.radius * cos(obj.angle);
        %     new_y = obj.center(2) + obj.radius * sin(obj.angle);
        %     obj.car_loc = [new_x, new_y, 0];
            
        %     % Update car orientation to be tangent to the circle
        %     obj.car_orientation = obj.angle + pi/2;
            
        %     % Update target location to match car location
        %     obj.target_loc = obj.car_loc;
            
        %     % Update time and step count
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
            constraints_satisfied = (obj.rate >= obj.R_min);
            
            % disp("this is the current step " + obj.stepCount);
            % disp("this is the current peb " + peb);
            if peb > 0.1
                base_reward = 1 / peb;
            else
                base_reward = 10 + (1.8 - 10) * (0.1 - peb) / 0.1;
            end
        
            if ~constraints_satisfied
                n = rand();
                reward = base_reward * n;
            else
                reward = base_reward;
            end
            reward = min(1.8, reward);
            % disp("this is the reward " + reward);
        end
        
        
        function state = reset(obj)
            % Reset simulation state
            obj.initializeChannels();
            obj.initializephi();
            obj.calculate_pathloss();
            [H_Los, H_Los_3d] = generate_H_Los(obj, obj.H_bt, obj.Nt, obj.Nb);
            [H_NLos, H_NLos_3d] = generate_H_NLoS(obj, obj.H_rt, obj.H_br, obj.Nt, obj.Nr, obj.Nb, obj.phi);
            obj.H_combined = obj.compute_Heff(H_Los, H_NLos);
            [obj.Wx,obj.W] = computeWx(obj);
            obj.Pb = obj.getpower();
            obj.gamma_c = obj.computeSNR();
            obj.rate = obj.getrate();
            obj.peb = obj.calculatePerformanceMetrics(obj.Wx, H_Los_3d, H_NLos_3d);
            obj.car_loc = obj.starting_pos;
            obj.target_loc = obj.car_loc;
            obj.time = 0;
            obj.stepCount = 0;
            obj.episodeCount = obj.episodeCount + 1;
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
        

        function [peb] = calculatePerformanceMetrics(obj, Wx, H_Los_3d, H_NLos_3d)
            [J, ~, ~] = computeFisherInformationMatrix(obj,Wx, H_Los_3d, H_NLos_3d);
            epsilon = 1e-6;
            J = (J + J') / 2;  % make Hermitian
            eigvals = eig(J);
            if min(eigvals) < 0
                J = J + epsilon * eye(size(J));  % regularize
            end
            CRLB = inv(J);
            obj.peb = sqrt(trace(CRLB));
            obj.peb = real(obj.peb);
            peb = obj.peb;
        end
        
        function [T] = computeTransformationMatrix(obj)
            T = zeros(2, 7);
            xb = obj.bs_loc(1);    yb = obj.bs_loc(2);    zb = obj.bs_loc(3);
            xr = obj.ris_loc(1);   yr = obj.ris_loc(2);   zr = obj.ris_loc(3);
            xt = obj.target_loc(1); yt = obj.target_loc(2);

            [~, L2, L3, ~, L_proj2, ~, ~, ~] = computeGeometricParameters(obj);

            % print(xb);
            % print(xr);
            % print()

            T(1,1) = (xt-xb) / (obj.c*L3);
            T(1,2) = (xt-xr) / (obj.c*L2);
            T(1,3) = (zr*(xt-xr)) / ((L2^3)*sqrt(1-((zr)^2)/(L2)^2));
            T(1,4) = ((yr-yt)*(xr-xt)) / ((L2^3)*sqrt(1-((yr-yt)^2)/(L2)^2));
            T(1,5) = (zr*(xr-xt)) / ((L2^3)*sqrt(1-((zr)^2)/(L2)^2));
            T(1,6) = (zb*(xt-xb)) / ((L3^3)*sqrt(1-(zb/L3)^2));
            T(1,7) = (zb*(xt-xb)) / ((L3^3)*sqrt(1-(zb/L3)^2));
            
            T(2,1) = (yt-yb) / (obj.c*L3);
            T(2,2) = (yt-yr) / (obj.c*L2);
            T(2,3) = (zr*(yt-yr)) / ((L2^3)*sqrt(1-((zr)^2)/(L2)^2));
            T(2,4) = ((L_proj2^2) + (yr-yt)*(xr-xt)) / ((L2^3)*sqrt(1-((yr-yt)^2)/(L2)^2));
            T(2,5) = (zr*(yr-yt)) / ((L2^3)*sqrt(1-((zr)^2)/(L2)^2));
            T(2,6) = (zb*(yt-yb)) / ((L3^3)*sqrt(1-(zb/L3)^2));
            T(2,7) = (zb*(yt-yb)) / ((L3^3)*sqrt(1-(zb/L3)^2));
        end

        % function [T] = computeTransformationMatrix(obj)
        %     % Extract locations
        %     xb = obj.bs_loc(1); yb = obj.bs_loc(2); zb = obj.bs_loc(3);
        %     xr = obj.ris_loc(1); yr = obj.ris_loc(2); zr = obj.ris_loc(3);
        %     xt = obj.target_loc(1); yt = obj.target_loc(2);
        
        %     % Get geometric values
        %     [~, L2, L3, ~, L_proj2, L_proj3, ~, ~] = computeGeometricParameters(obj);
        
        %     % Initialize T
        %     T = zeros(2, 7);
        
        %     % First row: partial derivatives w.r.t. x_t
        %     T(1,1) = (xt - xb) / (obj.c * L3);                                              % dτ_l/dx_t
        %     T(1,2) = (xt - xr) / (obj.c * L2);                                              % dτ_nl/dx_t
        %     T(1,3) = (zr * (xt - xr)) / (L2^3 * sqrt(1 - (zr^2) / L2^2));                   % dψ_rt/dx_t
        %     T(1,4) = ((yr - yt) * (xr - xt)) / (L2^3 * sqrt(1 - ((yr - yt)^2) / L_proj2^2));% dφ_rt^a/dx_t
        %     T(1,5) = (zr * (xr - xt)) / (L2^3 * sqrt(1 - (zr^2) / L2^2));                   % dφ_rt^e/dx_t
        %     T(1,6) = (zb * (xt - xb)) / (L3^3 * sqrt(1 - (L_proj3^2) / L3^2));              % dψ_bt/dx_t
        %     T(1,7) = (zb * (xt - xb)) / (L3^3 * sqrt(1 - (L_proj3^2) / L3^2));              % dψ_tb/dx_t
        
        %     % Second row: partial derivatives w.r.t. y_t
        %     T(2,1) = (yt - yb) / (obj.c * L3);                                              % dτ_l/dy_t
        %     T(2,2) = (yt - yr) / (obj.c * L2);                                              % dτ_nl/dy_t
        %     T(2,3) = (zr * (yt - yr)) / (L2^3 * sqrt(1 - (zr^2) / L2^2));                   % dψ_rt/dy_t
        %     T(2,4) = (L_proj2^2 + (yr - yt)*(xr - xt)) / (L2^3 * sqrt(1 - ((yr - yt)^2) / L_proj2^2)); % dφ_rt^d/dy_t
        %     T(2,5) = (zr * (yr - yt)) / (L2^3 * sqrt(1 - (zr^2) / L2^2));                   % dφ_rt^e/dy_t
        %     T(2,6) = (zb * (yt - yb)) / (L3^3 * sqrt(1 - (L_proj3^2) / L3^2));              % dψ_bt/dy_t
        %     T(2,7) = (zb * (yt - yb)) / (L3^3 * sqrt(1 - (L_proj3^2) / L3^2));              % dψ_tb/dy_t

        % end
        
        
        % ! Don't remove this
        function [Wx, W] = computeWx(obj)
            N = obj.Ns;
            if obj.stepCount > 1
                W_mrt = obj.H_combined;
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
        
        function [J, J_zao, T] = computeFisherInformationMatrix(obj, Wx ,H_Los_3d, H_NLos_3d)
            % sigma_s = sqrt(obj.Pb/obj.gamma_c); 
            format long g; % Noise variance (placeholder)
            [T] = computeTransformationMatrix(obj);
            % disp("this is T")
            % disp(T)
            J_zao = calculate_Jzao(obj, obj.Pb_dbm, obj.Ns, Wx, H_Los_3d, H_NLos_3d);
            % disp(J_zao);
            % disp(J_zao);
            % Compute final Fisher Information Matrix
            J = T * J_zao * conj(T');
            obj.J = J;
        end

        function [J_zao] = calculate_Jzao(obj, Pb_dbm, N, Wx, H_Los_3d, H_NLos_3d)
            % Initialize Fisher Information Matrix
            J_zao = zeros(7, 7);
            
            [~, ~, ~, ~, ~, ~, delays, angles] = computeGeometricParameters(obj);
            N=obj.Ns;
            tau_l = delays.line_of_sight;
            tau_nl = delays.non_line_of_sight;
            % Different derivative calculations for different parameters
            % For time delays (param_idx 1-2)
            psi_bt = angles.bs_to_target_transmit;
            psi_tb = angles.bs_to_target_receive;
            
            % BS-RIS channel parameters
            psi_br = angles.bs_to_ris.azimuth;
            phi_abr = angles.bs_to_ris.elevation_azimuth;
            phi_ebr = angles.bs_to_ris.elevation_angle;

            a_psi_br = compute_a_psi(obj, obj.Nb, psi_br, obj.lambda, obj.lambda/2);
            a_phi_abr = compute_a_phi(obj, sqrt(obj.Nr), phi_abr, phi_ebr, obj.lambda, obj.lambda/2);
            
            % RIS-Target channel parameters
            psi_rt = angles.ris_to_target.azimuth;
            phi_art = angles.ris_to_target.elevation_angle;
            phi_ert = angles.ris_to_target.elevation_angle;

            a_psi_rt = compute_a_psi(obj, obj.Nt, psi_rt, obj.lambda, obj.lambda/2);
            a_phi_art = compute_a_phi(obj, sqrt(obj.Nr), phi_art, phi_ert, obj.lambda/2, obj.lambda/2);

            % Get frequency-dependent steering vectors (dimensions: Nb×Ns and Nt×Ns)
            a_psi_bt = compute_a_psi(obj, obj.Nb, psi_bt, obj.lambda, obj.lambda/2);
            a_psi_tb = compute_a_psi(obj, obj.Nt, psi_tb, obj.lambda, obj.lambda/2);
            a_rt = 1j * (2*pi/obj.lambda) * cos(psi_rt) * diag(0:obj.Nt-1);
            a_bt = 1j * (2*pi/obj.lambda) * cos(psi_bt) * diag(0:obj.Nt-1);
            a_tb = 1j * (2*pi/obj.lambda) * cos(psi_tb) * diag(0:obj.Nt-1);

            gamma_l = sqrt(obj.Nb*obj.Nt)/sqrt(3);
            gamma_nl = sqrt(obj.Nb*obj.Nt)/sqrt(4);

            a_rt_a = 1j * (2 * pi / obj.lambda) * (obj.lambda/2) * ((obj.Nx - 1) * cos(phi_art) * sin(phi_ert));
            a_rt_e = 1j * (2 * pi / obj.lambda) * (obj.lambda/2) * ((obj.Nx - 1) * sin(phi_art) * cos(phi_ert) - (obj.Ny - 1) * sin(phi_ert));
            
            % Calculate scaling factor with normalization to prevent numerical issues
            scaling_factor = 1;
            % disp(scaling_factor);
            for i = 1:7
                for j = 1:7
                    sum_term = 0;
                    for n = 1:N
                        % mu = (H_Los_3d(:,:,n) + H_NLos_3d(:,:,n)) * Wx(:,n);
                        % disp(mu)
                        
                        % Use a more accurate derivative calculation
                        dmu_i = calculate_derivative(obj, i, n, Wx, a_psi_br, a_phi_abr, a_psi_rt, a_phi_art, a_psi_bt, a_psi_tb, a_rt, a_bt, a_tb, gamma_l, gamma_nl, a_rt_a, a_rt_e, tau_l, tau_nl);
                        dmu_j = calculate_derivative(obj, j, n, Wx, a_psi_br, a_phi_abr, a_psi_rt, a_phi_art, a_psi_bt, a_psi_tb, a_rt, a_bt, a_tb, gamma_l, gamma_nl, a_rt_a, a_rt_e, tau_l, tau_nl);
                        
                        dmu_i_h = conj(dmu_i');
                        inner_product = dmu_i_h * dmu_j;
                        sum_term = sum_term + real(inner_product);

                    end
                    
                    J_zao(i,j) = scaling_factor * sum_term;
                    % disp("for value of i: " + i);
                    % disp("for value of j:" + j);
                    % disp(J_zao(i,j));
                end
            end
            obj.J_zao = J_zao;
        end
        
        % Helper function for calculating derivatives
        function dmu = calculate_derivative(obj, param_idx, subcarrier_idx, Wx, a_psi_br, a_phi_abr, a_psi_rt, a_phi_art, a_psi_bt, a_psi_tb, a_rt, a_bt, a_tb, gamma_l, gamma_nl, a_rt_a, a_rt_e, tau_l, tau_nl)
            % This function should implement proper partial derivative calculation
            % based on the channel model and parameter typ
            N = 10;
            freq_factor = (subcarrier_idx / N);
            % A1 = gamma_l * obj.h_l * (1j * 2 * pi * obj.B * freq_factor) * exp(1j * 2 * pi * obj.B * freq_factor * tau_l);
            % A2 = gamma_nl * obj.h_nl * (1j * 2 * pi * obj.B * freq_factor) * exp(1j * 2 * pi * obj.B * freq_factor * tau_nl);
            

            if param_idx <= 2
                % Time domain derivative (frequency domain multiplication)
                % omega_n = 2*pi*subcarrier_idx; % Angular frequency
                if param_idx == 1 % LoS delay
                    A4 = gamma_l * obj.h_l * exp(1j * 2 * pi * obj.B * freq_factor * tau_l);
                    dmu =  A4 * a_psi_bt(:,subcarrier_idx) * conj(a_psi_tb(:,subcarrier_idx)') * Wx(:,subcarrier_idx);

                else % NLoS delay
                    A3 = gamma_nl * obj.h_nl * exp(1j * 2 * pi * obj.B * freq_factor * tau_nl);
                    dmu = A3 * a_psi_rt(:,subcarrier_idx) * conj(a_phi_art(:,subcarrier_idx)') * obj.phi * a_phi_abr(:,subcarrier_idx) * conj(a_psi_br(:,subcarrier_idx)') *...
                        Wx(:, subcarrier_idx);
                end
            % For angle parameters (param_idx 3-7)
            else
                % Different derivatives for different angle parameters
                % (This is a placeholder - actual calculations would be model-specific)
                if param_idx == 3 % psi_rt
                    A3 = gamma_nl * obj.h_nl * exp(1j * 2 * pi * obj.B * freq_factor * tau_nl);
                    dmu = A3 * ...
                        a_rt * a_psi_rt(:,subcarrier_idx) * conj(a_phi_art(:,subcarrier_idx)') * obj.phi * ...
                        a_phi_abr(:,subcarrier_idx) * conj(a_psi_br(:,subcarrier_idx)') * Wx(:, subcarrier_idx);
                elseif param_idx == 4 % phi_rt_a
                    A3 = gamma_nl * obj.h_nl * exp(1j * 2 * pi * obj.B * freq_factor * tau_nl);
                    dmu = A3 * ...
                        a_psi_rt(:,subcarrier_idx) * conj(a_phi_art(:,subcarrier_idx)') * diag(a_rt_a) * obj.phi * ...
                        a_phi_abr(:,subcarrier_idx) * conj(a_psi_br(:,subcarrier_idx)') * Wx(:, subcarrier_idx);
                elseif param_idx == 5 % phi_rt_e
                    A3 = gamma_nl * obj.h_nl * exp(1j * 2 * pi * obj.B * freq_factor * tau_nl);
                    dmu = A3 * ...
                        a_psi_rt(:,subcarrier_idx) * conj(a_phi_art(:,subcarrier_idx)') * diag(a_rt_e) * obj.phi * a_phi_abr(:,subcarrier_idx) * ...
                        conj(a_psi_br(:,subcarrier_idx)')*Wx(:,subcarrier_idx);
                elseif param_idx == 6 % psi_bt
                    A4 = gamma_l * obj.h_l * exp(1j * 2 * pi * obj.B * freq_factor * tau_l);
                    dmu = A4 * ...
                        a_psi_tb(:,subcarrier_idx) * ...
                        conj(a_psi_bt(:,subcarrier_idx)') * ...
                        a_bt * ...
                        a_psi_bt(:,subcarrier_idx) * ...
                        (a_psi_bt(:,subcarrier_idx)' * Wx(:, subcarrier_idx));

                else % psi_tb
                    A4 = gamma_l * obj.h_l * exp(1j * 2 * pi * obj.B * freq_factor * tau_l);
                    dmu = A4 * ...
                        a_psi_tb(:,subcarrier_idx) * ...
                        conj(a_psi_bt(:,subcarrier_idx)') * ...
                        a_tb * ...
                        a_psi_tb(:,subcarrier_idx) * ...
                        (a_psi_tb(:,subcarrier_idx)' * Wx(:, subcarrier_idx));                  
                end
            end
        end

        % function [J_zao] = calculate_Jzao(obj, Pb_dbm, N, Wx, H_Los_3d, H_NLos_3d)
        %     % Initialize Fisher Information Matrix
        %     J_zao = zeros(7, 7);
            
        %     % Compute geometric parameters once
        %     [~, ~, ~, ~, ~, ~, delays, angles] = computeGeometricParameters(obj);
            
        %     % Extract parameters
        %     psi_rt = angles.ris_to_target.aoa;
        %     psi_bt = angles.bs_to_target_transmit;
        %     psi_tb = angles.bs_to_target_receive;
        %     phi_rt_a = angles.ris_to_target.azimuth;
        %     phi_rt_e = angles.ris_to_target.elevation_angle;
        %     tau_l = delays.line_of_sight;
        %     tau_nl = delays.non_line_of_sight;
            
        %     zao = {tau_l, tau_nl, psi_rt, phi_rt_a, phi_rt_e, psi_bt, psi_tb};
            
        %     % Calculate scaling factor with normalization to prevent numerical issues
        %     scaling_factor = -1;
            
        %     % Precompute all derivatives for all parameters and subcarriers
        %     derivatives = cell(7, N);
        %     for i = 1:7
        %         for n = 1:N
        %             mu = (H_Los_3d(:,:,n) + H_NLos_3d(:,:,n)) * Wx(:,n);
        %             derivatives{i,n} = calculate_derivative(obj, mu, zao, i, n, H_Los_3d, H_NLos_3d, Wx);
        %         end
        %     end
            
        %     % Calculate FIM elements
        %     for i = 1:7
        %         for j = i:7  % Use symmetry of FIM to reduce computations
        %             sum_term = 0;
        %             for n = 1:N
        %                 dmu_i = derivatives{i,n};
        %                 dmu_j = derivatives{j,n};
                        
        %                 % Hermitian transpose
        %                 dmu_i_h = conj(dmu_i');
        %                 inner_product = dmu_i_h * dmu_j;
        %                 sum_term = sum_term + real(inner_product);
        %             end
                    
        %             J_zao(i,j) = scaling_factor * sum_term;
        %             J_zao(j,i) = J_zao(i,j);  % Use symmetry property
        %         end
        %     end
        %     % disp(J_zao);
        % end
        
        % function dmu = calculate_derivative(obj, param_idx, subcarrier_idx, Wx)
        %     % Extract parameters once
        %     [~, ~, ~, ~, ~, ~, delays, angles] = computeGeometricParameters(obj);
        %     N = obj.Ns;
        %     tau_l = delays.line_of_sight;
        %     tau_nl = delays.non_line_of_sight;
            
        %     % BS-Target angles
        %     psi_bt = angles.bs_to_target_transmit;
        %     psi_tb = angles.bs_to_target_receive;
            
        %     % BS-RIS channel parameters
        %     psi_br = angles.bs_to_ris.azimuth;
        %     phi_abr = angles.bs_to_ris.elevation_azimuth;
        %     phi_ebr = angles.bs_to_ris.elevation_angle;
            
        %     % RIS-Target channel parameters
        %     psi_rt = angles.ris_to_target.azimuth;
        %     phi_art = angles.ris_to_target.elevation_angle;
        %     phi_ert = angles.ris_to_target.elevation_angle;
            
        %     % Precompute frequency factor
        %     freq_factor = (subcarrier_idx / N);
        %     omega_n = 2 * pi * obj.B * freq_factor;
            
        %     % Precompute common factors
        %     gamma_l = sqrt(obj.Nb*obj.Nt)/sqrt(obj.rho_l);
        %     gamma_nl = sqrt(obj.Nb*obj.Nt)/sqrt(obj.rho_nl);
            
        %     exp_factor_los = exp(1j * omega_n * tau_l);
        %     exp_factor_nlos = exp(1j * omega_n * tau_nl);
            
        %     % Precompute all steering vectors needed by any case
        %     a_psi_bt = compute_a_psi(obj, obj.Nb, psi_bt, obj.lambda, obj.lambda/2);
        %     a_psi_tb = compute_a_psi(obj, obj.Nt, psi_tb, obj.lambda, obj.lambda/2);
        %     a_psi_rt = compute_a_psi(obj, obj.Nt, psi_rt, obj.lambda, obj.lambda/2);
        %     a_psi_br = compute_a_psi(obj, obj.Nb, psi_br, obj.lambda, obj.lambda/2);
        %     a_phi_abr = compute_a_phi(obj, sqrt(obj.Nr), phi_abr, phi_ebr, obj.lambda, obj.lambda/2);
        %     a_phi_art = compute_a_phi(obj, sqrt(obj.Nr), phi_art, phi_ert, obj.lambda/2, obj.lambda/2);
            
        %     % Different derivative calculations for different parameters
        %     switch param_idx
        %         case 1 % LoS delay
        %             % Time domain derivative (frequency domain multiplication)
        %             a_psi_bt_n = a_psi_bt(:,subcarrier_idx);
        %             a_psi_tb_n = a_psi_tb(:,subcarrier_idx);
                    
        %             dmu = gamma_l * obj.h_l * (1j * omega_n) * exp_factor_los * ...
        %                 a_psi_bt_n * conj(a_psi_tb_n') * Wx(:,subcarrier_idx);
                    
        %         case 2 % NLoS delay
        %             a_psi_rt_n = a_psi_rt(:,subcarrier_idx);
        %             a_phi_art_n = a_phi_art(:,subcarrier_idx);
        %             a_phi_abr_n = a_phi_abr(:,subcarrier_idx);
        %             a_psi_br_n = a_psi_br(:,subcarrier_idx);
                    
        %             dmu = gamma_nl * obj.h_nl * (1j * omega_n) * exp_factor_nlos * ...
        %                 a_psi_rt_n * conj(a_phi_art_n') * obj.phi * a_phi_abr_n * ...
        %                 conj(a_psi_br_n') * Wx(:, subcarrier_idx);
                    
        %         case 3 % psi_rt
        %             a_psi_rt_n = a_psi_rt(:,subcarrier_idx);
        %             a_phi_art_n = a_phi_art(:,subcarrier_idx);
        %             a_phi_abr_n = a_phi_abr(:,subcarrier_idx);
        %             a_psi_br_n = a_psi_br(:,subcarrier_idx);
                    
        %             % Precompute the steering vector derivative
        %             a_rt = 1j * (2*pi/obj.lambda) * cos(psi_rt) * diag(0:obj.Nt-1);
                    
        %             dmu = gamma_nl * obj.h_nl * exp_factor_nlos * ...
        %                 a_rt * a_psi_rt_n * conj(a_phi_art_n') * obj.phi * ...
        %                 a_phi_abr_n * conj(a_psi_br_n') * Wx(:, subcarrier_idx);
                    
        %         case 4 % phi_rt_a
        %             a_psi_rt_n = a_psi_rt(:,subcarrier_idx);
        %             a_phi_art_n = a_phi_art(:,subcarrier_idx);
        %             a_phi_abr_n = a_phi_abr(:,subcarrier_idx);
        %             a_psi_br_n = a_psi_br(:,subcarrier_idx);
                    
        %             % Angle derivative
        %             a_rt_a = 1j * (2 * pi / obj.lambda) * (obj.lambda/2) * ((obj.Nx - 1) * cos(phi_art) * sin(phi_ert));
                    
        %             dmu = gamma_nl * obj.h_nl * exp_factor_nlos * ...
        %                 a_psi_rt_n * conj(a_phi_art_n') * diag(a_rt_a) * obj.phi * ...
        %                 a_phi_abr_n * conj(a_psi_br_n') * Wx(:, subcarrier_idx);
                    
        %         case 5 % phi_rt_e
        %             a_psi_rt_n = a_psi_rt(:,subcarrier_idx);
        %             a_phi_art_n = a_phi_art(:,subcarrier_idx);
        %             a_phi_abr_n = a_phi_abr(:,subcarrier_idx);
        %             a_psi_br_n = a_psi_br(:,subcarrier_idx);
                    
        %             % Angle derivative
        %             a_rt_e = 1j * (2 * pi / obj.lambda) * (obj.lambda/2) * ((obj.Nx - 1) * sin(phi_art) * cos(phi_ert) - (obj.Ny - 1) * sin(phi_ert));
                    
        %             dmu = gamma_nl * obj.h_nl * exp_factor_nlos * ...
        %                 a_psi_rt_n * conj(a_phi_art_n') * diag(a_rt_e) * obj.phi * a_phi_abr_n * ...
        %                 conj(a_psi_br_n') * Wx(:, subcarrier_idx);
                    
        %         case 6 % psi_bt
        %             a_psi_bt_n = a_psi_bt(:,subcarrier_idx);
        %             a_psi_tb_n = a_psi_tb(:,subcarrier_idx);
                    
        %             % Angle derivative
        %             a_bt = 1j * (2*pi/obj.lambda) * cos(psi_bt) * diag(0:obj.Nt-1);
                    
        %             % Optimize matrix multiplications
        %             inner_product = a_psi_bt_n' * Wx(:, subcarrier_idx);
                    
        %             dmu = gamma_l * obj.h_l * exp_factor_los * ...
        %                 a_psi_tb_n * conj(a_psi_bt_n') * a_bt * a_psi_bt_n * inner_product;
                    
        %         case 7 % psi_tb
        %             a_psi_bt_n = a_psi_bt(:,subcarrier_idx);
        %             a_psi_tb_n = a_psi_tb(:,subcarrier_idx);
                    
        %             % Angle derivative
        %             a_tb = 1j * (2*pi/obj.lambda) * cos(psi_tb) * diag(0:obj.Nt-1);
                    
        %             % Optimize matrix multiplications
        %             inner_product = a_psi_tb_n' * Wx(:, subcarrier_idx);
                    
        %             dmu = gamma_l * obj.h_l * exp_factor_los * ...
        %                 a_psi_tb_n * conj(a_psi_bt_n') * a_tb * a_psi_tb_n * inner_product;
        %     end
        % end
        % ! -------------------- PEB COMPUTATION PART ENDS HERE --------------------        

    end
    
end
