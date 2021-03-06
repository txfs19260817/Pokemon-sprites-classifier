workers = 2
worker_class = "gevent"
bind = "0.0.0.0:14514"

loglevel = 'info'
accesslog = '-'
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s"'

certfile = "/www/server/panel/vhost/cert/scopelens.team/fullchain.pem"
keyfile = "/www/server/panel/vhost/cert/scopelens.team/privkey.pem"