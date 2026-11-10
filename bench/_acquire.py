import fcntl, sys, time
f = open("/tmp/mojo-bench.lock", "a+")
t = time.time()
while True:
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        print("ACQUIRED after %.0fs at %s" % (time.time() - t, time.strftime("%H:%M:%SZ", time.gmtime())), flush=True)
        sys.exit(0)
    except BlockingIOError:
        time.sleep(0.5)
