% System Parameters
Nb = 4;         % Number of BS antennas (ULA)
Nt = 4;         % Number of Vehicle antennas
Nx = 8;         % Number of RIS elements in x-direction
Ny = 8;         % Number of RIS elements in y-direction
Nr = Nx * Ny;   % Total number of RIS elements
B = 20e6;         % MHz
fc = 28e9;
c = physconst('lightspeed');
lambda = c/fc;
d = 0.5*lambda;
% dr = 0.5*lambda;
% dc = 0.5*lambda;

% Initialize parameters
x_size = 1000;  % Environment size in meters
y_size = 1000;
z_size = 100;   % Added height dimension

% Initial positions
vehicle_pos = [500, 500, 0];  % Vehicle starts at ground level
goal_pos = [1000, 500, 0];    % Goal position
bs_loc = [900, 100, 20];      % Base station location
ris_loc = [200, 300, 40];     % RIS location

% Create ULA positions for BS (aligned along y-axis)
bs_array_positions = zeros(Nb, 3);
for i = 1:Nb
    bs_array_positions(i,:) = bs_loc + [0, (i-1)*d, 0];
end

% Create ULA positions for Vehicle (aligned along y-axis)
vehicle_array_positions = zeros(Nt, 3);
% Create UPA positions for RIS (in x-y plane)
ris_array_positions = zeros(Nr, 3);
idx = 1;
for ix = 1:Nx
    for iy = 1:Ny
        ris_array_positions(idx,:) = ris_loc + [(ix-1)*d, (iy-1)*d, 0];
        idx = idx + 1;
    end
end


function [H_bt, H_br, H_rt] = generate_channels(Nt, Nr, Nb)
    % Constants
    lambda = 3e8/2e7;  % wavelength
    d = lambda/2;      % antenna spacing
    dr = lambda/2;     % element spacing for 2D arrays
    
    % Get geometric parameters
    [~,~,~,~,~,~,~,angles] = computeGeometricParameters();
    
    % Generate BS-Target channel
    H_bt = generate_H_bt(Nt, Nr, angles, lambda, d);
    
    % Generate BS-RIS and RIS-Target channels
    [H_br, H_rt] = generate_H_br_H_rt(Nb, Nr, Nt, angles, lambda, d, dr);
end

function H_bt = generate_H_bt(Nt, Nr, angles, lambda, d)
    psi_bt = angles.bs_to_target.aoa_in;
    psi_tb = angles.bs_to_target.aoa_out;
    
    a_psi_bt = generate_steering_vector(Nr, psi_bt, lambda, d);
    a_psi_tb = generate_steering_vector(Nt, psi_tb, lambda, d);
    
    H_bt = a_psi_tb * a_psi_bt';
end

function [H_br, H_rt] = generate_H_br_H_rt(Nb, Nr, Nt, angles, lambda, d, dr)
    % BS-RIS channel parameters
    psi_br = angles.bs_to_ris.azimuth;
    phi_abr = angles.bs_to_ris.elevation_azimuth;
    phi_ebr = angles.bs_to_ris.elevation_angle;
    
    % RIS-Target channel parameters
    psi_rt = angles.ris_to_target.azimuth;
    phi_art = angles.ris_to_target.elevation_angle;
    phi_ert = angles.ris_to_target.elevation_angle;
    
    % Generate channels
    H_br = generate_H_br(Nr, Nb, phi_abr, phi_ebr, psi_br, lambda, dr, d);
    H_rt = generate_H_rt(Nt, Nr, phi_art, phi_ert, psi_rt, lambda, dr, d);
end

function H_br = generate_H_br(Nr, Nb, phi_abr, phi_ebr, psi_br, lambda, dr, d)
    a_psi_br = generate_steering_vector(Nb, psi_br, lambda, d);
    a_phi_abr = generate_2d_steering_vector(sqrt(Nr), phi_abr, phi_ebr, lambda, dr);
    
    H_br = a_phi_abr * a_psi_br';
end

function H_rt = generate_H_rt(Nt, Nr, phi_art, phi_ert, psi_rt, lambda, dr, d)
    a_psi_rt = generate_steering_vector(Nt, psi_rt, lambda, d);
    a_phi_art = generate_2d_steering_vector(sqrt(Nr), phi_art, phi_ert, lambda, dr);
    
    H_rt = a_psi_rt * a_phi_art';
end

function a_vec = generate_steering_vector(Nant, psi, lambda, d)
    k = 2*pi/lambda;
    n = 0:(Nant-1);
    phase_terms = exp(1j * k * d * n * sin(psi));
    a_vec = phase_terms(:) / sqrt(Nant);
end

function a_phi = generate_2d_steering_vector(Nx, phi_a, phi_e, lambda, dr)
    N2 = Nx * Nx;
    a_phi = zeros(N2, 1);
    k = 2*pi/lambda;
    
    idx = 1;
    for m = 1:Nx
        for n = 1:Nx
            phase_term = exp(1j * k * dr * (m * sin(phi_a) * sin(phi_e) + n * cos(phi_e)));
            a_phi(idx) = phase_term;
            idx = idx + 1;
        end
    end
    
    a_phi = a_phi / sqrt(N2);
end

function [H_NLos, H_Los, gamma_c] = simulate_downlink_transmission(~, ~, ~, W)
    % System Parameters
    Nb = 4;      % Number of BS antennas (ULA)
    Nt = 4;      % Number of Vehicle antennas
    Nx = 8;      % Number of RIS elements in x-direction
    Ny = 8;      % Number of RIS elements in y-direction
    Nr = Nx * Ny; % Total number of RIS elements
    Mb = 64;     % Number of subcarriers
    
    % Generate transmit data for each subcarrier
    x = (randn(Mb,1) + 1i*randn(Mb,1))/sqrt(2);
    
    % Generate beamforming matrix if not provided
    if nargin < 4
        W = zeros(Nb, Mb);
        for i = 1:Mb
            w = (randn(Nb,1) + 1i*randn(Nb,1))/sqrt(2*Nb);
            W(:,i) = w/norm(w); % Normalize each beamforming vector
        end
    end
    
    % Verify transmit power constraint
    % Pb = norm(W*x)^2;

    [H_bt, H_br, H_rt] = generate_channels(Nt, Nr, Nb);
    
    H_Los = generate_H_Los(H_bt, Nt, Nr, Nb);

    H_NLos = generate_H_NLoS(H_rt, H_br, Nt, Nr, Nb);

    gamma_c = 90;
end

function H_Los = generate_H_Los(H_bt, Nt, Nr, Nb)
    % Parameters
    K_dB = 4; % Rician K-factor in dB
    K = 10^(K_dB/10);
    sigma = sqrt(1/(2*(K+1)));
    mu = sqrt(K/(K+1));
    
    % Generate small-scale fading
    hl = (sigma * complex(randn(1,1), randn(1,1))) + mu;
    
    % Path loss in linear scale (110 dB)
    rho_l = 3;
    
    % Calculate gamma_l
    gamma_l = sqrt(Nb*Nt)/sqrt(rho_l);
    
    % System parameters
    B = 20e6; % Bandwidth (20 MHz)
    N = 2048; % FFT size
    [~,~,~,~, ~, ~, delays, ~] = computeGeometricParameters();
    tau_l = delays.line_of_sight;
    
    % Generate H_Los for all subcarriers with correct dimensions Nt × Nr
    H_Los = zeros(Nt, Nr, N);
    
    for n = 1:N
        phase = exp(1j*2*pi*B*(n-1)*tau_l/N);
        % Need to adjust this multiplication to get Nt × Nr result
        H_Los(:,:,n) = gamma_l * hl * H_bt * phase;  
    end
end

function H_NLoS = generate_H_NLoS(H_rt, H_br, Nt, Nr, Nb)
    % Parameters
    K_dB = 4; % Rician K-factor in dB
    K = 10^(K_dB/10);
    sigma = sqrt(1/(2*(K+1)));
    mu = sqrt(K/(K+1));
    
    % Generate small-scale fading
    hnl = (sigma * complex(randn(1,1), randn(1,1))) + mu;
    
    % Path loss in linear scale
    rho_nl = 4;
    
    % Calculate gamma_nl
    gamma_nl = sqrt(Nb*Nr)/sqrt(rho_nl);
    
    % System parameters
    B = 20e6; % Bandwidth (20 MHz)
    N = 2048; % FFT size
    [~,~,~,~, ~, ~, delays, ~] = computeGeometricParameters();
    tau_nl = delays.non_line_of_sight;
    
    % Generate RIS reflection parameters (u)
    rho_r = 1;
    theta = 2*pi*rand(Nr,1);
    u = rho_r * exp(1j*theta);
    Phi = diag(u);
    
    % Generate H_NLoS for all subcarriers
    H_NLoS = zeros(Nt, Nb, N);  % Changed to Nt × Nr
    
    for n = 1:N
        phase = exp(1j*2*pi*B*(n-1)*tau_nl/N);
        % H_rt(Nt×Nr) * Phi(Nr×Nr) * H_br(Nr×Nb)
        H_NLoS(:,:,n) = gamma_nl * hnl * H_rt * Phi * H_br * phase;
    end
end

function [L1, L2, L3, L_proj1, L_proj2, L_proj3, delays, angles] = computeGeometricParameters()
    
    target_loc = [500, 500, 0];  % Vehicle starts at ground level
    bs_loc = [900, 100, 20];      % Base station location
    ris_loc = [200, 300, 40]; 
    % Extract coordinates
    xb = bs_loc(1);    yb = bs_loc(2);    zb = bs_loc(3);
    xr = ris_loc(1);   yr = ris_loc(2);   zr = ris_loc(3);
    xt = target_loc(1); yt = target_loc(2); 
    % zt = target_loc(3);
    
    % Speed of light
    c = 3e8;
    
    % Calculate 3D Euclidean distances
    L1 = sqrt((xb - xr)^2 + (yb - yr)^2 + (zb - zr)^2);
    L2 = sqrt((xr - xt)^2 + (yr - yt)^2 + zr^2);
    L3 = sqrt((xb - xt)^2 + (yb - yt)^2 + zb^2);
    
    % Calculate 2D (X-Y plane) projections
    L_proj1 = sqrt((xb - xr)^2 + (yb - yr)^2);
    L_proj2 = sqrt((xr - xt)^2 + (yr - yt)^2);
    L_proj3 = sqrt((xb - xt)^2 + (yb - yt)^2);
    
    % Calculate signal delays
    delays.line_of_sight = L3 / c;
    delays.non_line_of_sight = (L1 + L2) / c;
    
    % Calculate angles
    % BS to RIS angles
    angles.bs_to_ris.azimuth = asin((zb - zr) / L1);
    angles.bs_to_ris.elevation_azimuth = asin((xb - xr) / L_proj1);
    angles.bs_to_ris.elevation_angle = acos((zb - zr) / L1);
    
    % RIS to Target angles
    angles.ris_to_target.aoa = asin(zr / L2);
    angles.ris_to_target.azimuth = acos((yr - yt) / L_proj2);
    angles.ris_to_target.elevation_angle = acos(zr / L2);

    % BS to Target angles
    angles.bs_to_target.aoa_in = acos(zb/L3);
    angles.bs_to_target.aoa_out = asin(zb/L3);
end


% Main simulation loop
time = 0;
idx = 1;
% Initialize variables
dt = 0.1;  % Time step
time = 0;
idx = 1;
speed = 10;  % Vehicle speed (modify as needed)

% Initialize storage for channel metrics
max_steps = ceil((goal_pos(1) - vehicle_pos(1))/(speed * dt));
channel_metrics = struct();
channel_metrics.time = zeros(1, max_steps);
channel_metrics.los_gain = zeros(1, max_steps);
channel_metrics.nlos_gain = zeros(1, max_steps);
channel_metrics.gamma = zeros(1, max_steps);

% Initialize beamforming matrix W (if needed)
Nb = 4;  % Number of BS antennas
Mb = 64; % Number of subcarriers
W = zeros(Nb, Mb);
for i = 1:Mb
    w = (randn(Nb,1) + 1i*randn(Nb,1))/sqrt(2*Nb);
    W(:,i) = w/norm(w);
end

% Create figure
figure('Name', '3D Vehicle Simulator with Antenna Arrays');
hold on;
grid on;
axis([0 x_size 0 y_size 0 z_size]);
xlabel('X Position (m)');
ylabel('Y Position (m)');
zlabel('Z Position (m)');
title('3D Vehicle Motion Simulation with Antenna Arrays');

% Plot BS ULA and center
bs_array = plot3(bs_array_positions(:,1), bs_array_positions(:,2), bs_array_positions(:,3), ...
    'ks', 'MarkerSize', 8, 'MarkerFaceColor', 'k');
bs_center = plot3(bs_loc(1), bs_loc(2), bs_loc(3), ...
    'ks', 'MarkerSize', 12, 'MarkerFaceColor', 'r');

% Plot RIS UPA and center
ris_array = plot3(ris_array_positions(:,1), ris_array_positions(:,2), ris_array_positions(:,3), ...
    'md', 'MarkerSize', 6, 'MarkerFaceColor', 'm');
ris_center = plot3(ris_loc(1), ris_loc(2), ris_loc(3), ...
    'md', 'MarkerSize', 12, 'MarkerFaceColor', 'r');

% Plot vehicle array
vehicle_array_markers = plot3(vehicle_array_positions(:,1), vehicle_array_positions(:,2), ...
    vehicle_array_positions(:,3), 'ro', 'MarkerSize', 8);

% Plot start and goal positions
start_pos = plot3(vehicle_pos(1), vehicle_pos(2), vehicle_pos(3), ...
    'go', 'MarkerSize', 10, 'MarkerFaceColor', 'g');
goal_pos_marker = plot3(goal_pos(1), goal_pos(2), goal_pos(3), ...
    'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'r');

% Plot connection lines
vehicle_to_bs = plot3([vehicle_pos(1), bs_loc(1)], ...
    [vehicle_pos(2), bs_loc(2)], ...
    [vehicle_pos(3), bs_loc(3)], 'b--');
vehicle_to_ris = plot3([vehicle_pos(1), ris_loc(1)], ...
    [vehicle_pos(2), ris_loc(2)], ...
    [vehicle_pos(3), ris_loc(3)], 'k--');

% Add legend with all components
legend([bs_array, bs_center, ris_array, ris_center, start_pos, goal_pos_marker, vehicle_array_markers, vehicle_to_bs, vehicle_to_ris], ...
    'BS Array', 'BS Center', 'RIS Array', 'RIS Center', 'Start', 'Goal', 'Vehicle Array', ...
    'Vehicle-BS Link', 'Vehicle-RIS Link');

% Main simulation loop
while vehicle_pos(1) < goal_pos(1)
    % Update vehicle center position
    vehicle_pos(1) = vehicle_pos(1) + speed * dt * cos(direction_angle);
    vehicle_pos(2) = vehicle_pos(2) + speed * dt * sin(direction_angle);
    vehicle_pos(3) = vehicle_pos(3) + speed * dt * sin(elevation_angle);
    time = time + dt;
    
    for i = 1:Nt
        vehicle_array_positions(i,:) = vehicle_pos + [0, (i-1)*d*cos(elevation_angle), (i-1)*d*sin(elevation_angle)];
    end
    
    % Calculate channels using new function
    [H_NLos, H_Los, gamma_c] = simulate_downlink_transmission([], [], [], W);
    
    % Store channel metrics
    channel_metrics.time(idx) = time;
    channel_metrics.los_gain(idx) = 20*log10(norm(H_Los, 'fro'));
    channel_metrics.nlos_gain(idx) = 20*log10(norm(H_NLos, 'fro'));
    channel_metrics.gamma(idx) = gamma_c;
    
    % Update vehicle visualization
    set(vehicle_array_markers, 'XData', vehicle_array_positions(:,1), ...
                              'YData', vehicle_array_positions(:,2), ...
                              'ZData', vehicle_array_positions(:,3));
    
    % Update connection lines
    set(vehicle_to_bs, 'XData', [vehicle_pos(1), bs_loc(1)], ...
                      'YData', [vehicle_pos(2), bs_loc(2)], ...
                      'ZData', [vehicle_pos(3), bs_loc(3)]);
    set(vehicle_to_ris, 'XData', [vehicle_pos(1), ris_loc(1)], ...
                       'YData', [vehicle_pos(2), ris_loc(2)], ...
                       'ZData', [vehicle_pos(3), ris_loc(3)]);
    
    % Update channel plots
    if mod(idx, 10) == 0
        figure(2);
        subplot(2,1,1);
        plot(channel_metrics.time(1:idx), channel_metrics.los_gain(1:idx), 'b-', ...
             channel_metrics.time(1:idx), channel_metrics.nlos_gain(1:idx), 'r-');
        grid on;
        xlabel('Time (s)');
        ylabel('Channel Gain (dB)');
        legend('LOS Channel', 'NLOS Channel');
        title('Channel Gains vs. Time');
        
        subplot(2,1,2);
        plot(channel_metrics.time(1:idx), channel_metrics.gamma(1:idx), 'g-');
        grid on;
        xlabel('Time (s)');
        ylabel('Gamma (degrees)');
        title('Phase Shift vs. Time');
    end
    
    % Force MATLAB to draw the update
    drawnow;
    
    % Add small pause to make motion visible
    pause(dt);
    
    idx = idx + 1;
end

% Trim any unused entries in the metrics
valid_indices = 1:(idx-1);
channel_metrics.time = channel_metrics.time(valid_indices);
channel_metrics.los_gain = channel_metrics.los_gain(valid_indices);
channel_metrics.nlos_gain = channel_metrics.nlos_gain(valid_indices);
channel_metrics.gamma = channel_metrics.gamma(valid_indices);

% Final plotting
figure(3);
subplot(2,1,1);
plot(channel_metrics.time, channel_metrics.los_gain, 'b-', ...
     channel_metrics.time, channel_metrics.nlos_gain, 'r-');
grid on;
xlabel('Time (s)');
ylabel('Channel Gain (dB)');
legend('LOS Channel', 'NLOS Channel');
title('Channel Gains vs. Time');

subplot(2,1,2);
plot(channel_metrics.time, channel_metrics.gamma, 'g-');
grid on;
xlabel('Time (s)');
ylabel('Gamma (degrees)');
title('Phase Shift vs. Time');

[received_signal, direct_channel, cascaded_channel] = simulate_downlink_transmission(bs_loc, ris_loc, vehicle_pos);