%C.3 s1

u = @(t) 1.0*(t>=0);
x = @(t) (u(t)-u(t-3)).*sin(5*t);
h = @(t) exp(t/5).*(u(t)-u(t-20));
dtau = 0.05; 
tau = 0:dtau:20;
ti = 0; 
tvec = 0:.1:20;
y2 = NaN*zeros(1,length(tvec)); % Pre-allocate memory
for t = tvec,
    ti = ti+1; % Time index
    xh = x(t-tau).*h(tau); 
    lxh = length(xh);        
    y2(ti) = sum(xh.*dtau); % Trapezoidal approximation of convolution integral
    subplot(2,1,1),plot(tau,h(tau),'k-',tau,x(t-tau),'k--',t,0,'ok');
    axis([tau(1) tau(end) -2 30]);grid;
    patch([tau(1:end-1);tau(1:end-1);tau(2:end);tau(2:end)],...
        [zeros(1,lxh-1);xh(1:end-1);xh(2:end);zeros(1,lxh-1)],...
        [.8 .8 .8],'edgecolor','none');
    xlabel('\tau'); title('h(\tau) [solid], x(t-\tau) [dashed], h(\tau)x(t-\tau) [gray]');
    c = get(gca,'children'); 
    set(gca,'children',[c(2);c(3);c(4);c(1)]);
    subplot(2,1,2),plot(tvec,y2,'k',tvec(ti),y2(ti),'ok');
    xlabel('t');     
    ylabel('y(t) = \int h(\tau)x(t-\tau) d\tau');
    axis([tau(1) tau(end) -5 5]); grid;
    drawnow;
end

%c.3 s2
u = @(t) 1.0*(t>=0);
x = @(t) (u(t)-u(t-3)).*sin(5*t);
h = @(t) 4*exp(-t/5).*(u(t)-u(t-20));
dtau = 0.05; 
tau = 0:dtau:20;
ti = 0; 
tvec = 0:.1:20;
y2 = NaN*zeros(1,length(tvec)); % Pre-allocate memory
for t = tvec,
    ti = ti+1; % Time index
    xh = x(t-tau).*h(tau); 
    lxh = length(xh);        
    y2(ti) = sum(xh.*dtau); % Trapezoidal approximation of convolution integral
    subplot(2,1,1),plot(tau,h(tau),'k-',tau,x(t-tau),'k--',t,0,'ok');
    axis([tau(1) tau(end) -2 10]);grid;
    patch([tau(1:end-1);tau(1:end-1);tau(2:end);tau(2:end)],...
        [zeros(1,lxh-1);xh(1:end-1);xh(2:end);zeros(1,lxh-1)],...
        [.8 .8 .8],'edgecolor','none');
    xlabel('\tau'); title('h(\tau) [solid], x(t-\tau) [dashed], h(\tau)x(t-\tau) [gray]');
    c = get(gca,'children'); 
    set(gca,'children',[c(2);c(3);c(4);c(1)]);
    subplot(2,1,2),plot(tvec,y2,'k',tvec(ti),y2(ti),'ok');
    xlabel('t');     
    ylabel('y(t) = \int h(\tau)x(t-\tau) d\tau');
    axis([tau(1) tau(end) -5 5]); grid;
    drawnow;
end

%C.3 s3
u = @(t) 1.0*(t>=0);
x = @(t) (u(t)-u(t-3)).*sin(5*t);
h = @(t) 4*exp(-t).*(u(t)-u(t-20));
dtau = 0.05; 
tau = 0:dtau:20;
ti = 0; 
tvec = 0:.1:20;
y2 = NaN*zeros(1,length(tvec)); % Pre-allocate memory
for t = tvec,
    ti = ti+1; % Time index
    xh = x(t-tau).*h(tau); 
    lxh = length(xh);        
    y2(ti) = sum(xh.*dtau); % Trapezoidal approximation of convolution integral
    subplot(2,1,1),plot(tau,h(tau),'k-',tau,x(t-tau),'k--',t,0,'ok');
    axis([tau(1) tau(end) -2.0 10]);grid;
    patch([tau(1:end-1);tau(1:end-1);tau(2:end);tau(2:end)],...
        [zeros(1,lxh-1);xh(1:end-1);xh(2:end);zeros(1,lxh-1)],...
        [.8 .8 .8],'edgecolor','none');
    xlabel('\tau'); title('h(\tau) [solid], x(t-\tau) [dashed], h(\tau)x(t-\tau) [gray]');
    c = get(gca,'children'); 
    set(gca,'children',[c(2);c(3);c(4);c(1)]);
    subplot(2,1,2),plot(tvec,y2,'k',tvec(ti),y2(ti),'ok');
    xlabel('t');     
    ylabel('y(t) = \int h(\tau)x(t-\tau) d\tau');
    axis([tau(1) tau(end) -5 5]); grid;
    drawnow;
end

%C.3 s4
u = @(t) 1.0*(t>=0);
x = @(t) (u(t)-u(t-3)).*sin(5*t);
h = @(t) 4*(exp(-t/5)-exp(-t)).*(u(t)-u(t-20));
dtau = 0.05; 
tau = 0:dtau:20;
ti = 0; 
tvec = 0:.1:20;
y2 = NaN*zeros(1,length(tvec)); % Pre-allocate memory
for t = tvec,
    ti = ti+1; % Time index
    xh = x(t-tau).*h(tau); 
    lxh = length(xh);        
    y2(ti) = sum(xh.*dtau); % Trapezoidal approximation of convolution integral
    subplot(2,1,1),plot(tau,h(tau),'k-',tau,x(t-tau),'k--',t,0,'ok');
    axis([tau(1) tau(end) -2.0 10]);grid;
    patch([tau(1:end-1);tau(1:end-1);tau(2:end);tau(2:end)],...
        [zeros(1,lxh-1);xh(1:end-1);xh(2:end);zeros(1,lxh-1)],...
        [.8 .8 .8],'edgecolor','none');
    xlabel('\tau'); title('h(\tau) [solid], x(t-\tau) [dashed], h(\tau)x(t-\tau) [gray]');
    c = get(gca,'children'); 
    set(gca,'children',[c(2);c(3);c(4);c(1)]);
    subplot(2,1,2),plot(tvec,y2,'k',tvec(ti),y2(ti),'ok');
    xlabel('t');     
    ylabel('y(t) = \int h(\tau)x(t-\tau) d\tau');
    axis([tau(1) tau(end) -5 5]); grid;
    drawnow;
end
