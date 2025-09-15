CREATE DATABASE fraud_db;

USE fraud_db;

CREATE TABLE single_checks (
    id INT AUTO_INCREMENT PRIMARY KEY,
    amount DOUBLE,
    time DOUBLE,
    hour INT,
    prediction VARCHAR(10),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE batch_predictions (
    id INT AUTO_INCREMENT PRIMARY KEY,
    txn_id VARCHAR(50),
    amount DOUBLE,
    time DOUBLE,
    hour INT,
    prediction VARCHAR(10),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
