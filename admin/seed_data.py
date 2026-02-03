#!/usr/bin/env python3
"""Seed sample data for admin app (DB tables + sample students + redis entries)"""
from structure import services

if __name__ == '__main__':
    print('Seeding DB and Redis with sample data...')
    services.seed_sample_data()
    print('Done.')
