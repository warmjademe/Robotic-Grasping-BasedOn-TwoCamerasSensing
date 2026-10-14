#!/usr/bin/env python3
"""Huashuo-only command typing in the existing ABB QEMU console, no public VNC."""
import argparse
import struct
import subprocess
import time
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--command');p.add_argument('--click',nargs=2,type=int,action='append');a=p.parse_args()
    if not a.command and not a.click:p.error('Provide --command or --click')
    child=subprocess.Popen(['docker','exec','-i','ABB','nc','127.0.0.1','5900'],stdin=subprocess.PIPE,stdout=subprocess.PIPE)
    def read(n):
        data=b''
        while len(data)<n:
            chunk=child.stdout.read(n-len(data))
            if not chunk:raise RuntimeError('Console closed')
            data+=chunk
        return data
    def send(x):child.stdin.write(x);child.stdin.flush()
    try:
        assert read(12).startswith(b'RFB 003.')
        send(b'RFB 003.008\n');n=read(1)[0];types=read(n);assert 1 in types
        send(b'\x01');assert read(4)==b'\0'*4
        send(b'\x01');header=read(24);read(struct.unpack('>I',header[-4:])[0])
        def key(code,down):send(struct.pack('>BBHI',4,int(down),0,code))
        if a.click:
            for x,y in a.click:
                assert 0<=x<1280 and 0<=y<800
                send(struct.pack('>BBHH',5,1,x,y));send(struct.pack('>BBHH',5,0,x,y));time.sleep(.5)
        if not a.command:return
        key(0xffeb,True);key(ord('r'),True);key(ord('r'),False);key(0xffeb,False);time.sleep(.8)
        key(0xffe3,True);key(ord('a'),True);key(ord('a'),False);key(0xffe3,False)
        shifted=dict(zip('~!@#$%^&*()_+{}|:"<>?','`1234567890-=[]\\;\',./'))
        for char in a.command:
            need_shift=char.isupper() or char in shifted
            base=shifted.get(char,char.lower() if char.isupper() else char)
            if need_shift:key(0xffe1,True)
            key(ord(base),True);key(ord(base),False)
            if need_shift:key(0xffe1,False)
            time.sleep(.004)
        key(0xff0d,True);key(0xff0d,False)
        time.sleep(.4)
    finally:
        child.terminate();child.wait(timeout=5)


if __name__=='__main__':main()
