-- MySQL Schema for Synthetic Training Data Tables
-- Based on analysis of synthetic_train_SMOTE_raw.csv and synthetic_train_ADASYN_raw.csv
-- Both tables have identical structure: 38 columns, all numeric (25 integers, 13 floats)
-- SMOTE: 482,668 rows
-- ADASYN: 482,815 rows

-- Drop tables if they exist (for clean setup)
DROP TABLE IF EXISTS synthetic_train_smote;
DROP TABLE IF EXISTS synthetic_train_adasyn;

-- Create synthetic_train_smote table
CREATE TABLE synthetic_train_smote (
    -- Auto-increment primary key (no natural unique key exists)
    id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    
    -- User Information
    user_id INT UNSIGNED NOT NULL,  -- Range: 100005-148544, 47,355 unique values
    
    -- Demographics
    age TINYINT UNSIGNED NOT NULL,  -- Range: 0-7, 8 unique values
    gender TINYINT UNSIGNED NOT NULL,  -- Range: 0-2, 3 unique values
    residence TINYINT UNSIGNED NOT NULL,  -- Range: 0-34, 35 unique values
    city SMALLINT UNSIGNED NOT NULL,  -- Range: 0-324, 325 unique values
    city_rank TINYINT UNSIGNED NOT NULL,  -- Range: 0-4, 5 unique values
    
    -- Device Information
    series_dev TINYINT UNSIGNED NOT NULL,  -- Range: 0-23, 24 unique values
    series_group TINYINT UNSIGNED NOT NULL,  -- Range: 0-7, 8 unique values
    emui_dev TINYINT UNSIGNED NOT NULL,  -- Range: 0-23, 24 unique values
    device_name TINYINT UNSIGNED NOT NULL,  -- Range: 0-209, 210 unique values
    device_size SMALLINT UNSIGNED NOT NULL,  -- Range: 0-592, 593 unique values
    net_type TINYINT UNSIGNED NOT NULL,  -- Range: 0-6, 7 unique values
    
    -- Advertisement Information
    task_id SMALLINT UNSIGNED NOT NULL,  -- Range: 0-4693, 4,694 unique values
    adv_id SMALLINT UNSIGNED NOT NULL,  -- Range: 0-5205, 5,205 unique values
    creat_type_cd TINYINT UNSIGNED NOT NULL,  -- Range: 0-9, 10 unique values
    adv_prim_id SMALLINT UNSIGNED NOT NULL,  -- Range: 0-408, 409 unique values
    inter_type_cd TINYINT UNSIGNED NOT NULL,  -- Range: 0-3, 4 unique values
    slot_id TINYINT UNSIGNED NOT NULL,  -- Range: 0-54, 55 unique values
    site_id TINYINT UNSIGNED NOT NULL,  -- Range: 0-1, 2 unique values
    spread_app_id TINYINT UNSIGNED NOT NULL,  -- Range: 0-88, 89 unique values
    hispace_app_tags TINYINT UNSIGNED NOT NULL,  -- Range: 0-36, 37 unique values
    app_second_class TINYINT UNSIGNED NOT NULL,  -- Range: 0-20, 21 unique values
    
    -- Floating Point Columns
    app_score DECIMAL(10,2) NOT NULL,  -- Range: 0.00-10.00, Mean: 8.40, Median: 10.00
    u_refreshTimes DECIMAL(10,2) NOT NULL,  -- Range: 0.00-9.00, Mean: 4.82, Median: 5.00
    u_feedLifeCycle DECIMAL(10,2) NOT NULL,  -- Range: 0.00-17.00, Mean: 15.90, Median: 17.00
    
    -- Time Features (all values are -1, indicating missing/unknown)
    hour TINYINT NOT NULL DEFAULT -1,  -- All values: -1 (missing timestamp data)
    dayofweek TINYINT NOT NULL DEFAULT -1,  -- All values: -1 (missing timestamp data)
    
    -- Feeds Domain Features (aggregated user behavior)
    feeds_u_phonePrice DECIMAL(10,2) NOT NULL,  -- Range: 0.00-6.00, Mean: 1.99, Median: 1.00
    feeds_u_browserLifeCycle DECIMAL(10,2) NOT NULL,  -- Range: 0.00-7.00, Mean: 4.10, Median: 6.00
    feeds_u_browserMode DECIMAL(10,2) NOT NULL,  -- Range: 0.00-4.00, Mean: 1.59, Median: 1.71
    feeds_u_feedLifeCycle DECIMAL(10,2) NOT NULL,  -- Range: 0.00-7.00, Mean: 3.98, Median: 5.58
    feeds_u_refreshTimes DECIMAL(10,2) NOT NULL,  -- Range: 0.00-9.00, Mean: 3.64, Median: 3.16
    feeds_u_newsCatInterests DECIMAL(10,2) NOT NULL,  -- Range: 0.00-4.00, Mean: 2.38, Median: 3.72
    feeds_u_newsCatDislike DECIMAL(10,2) NOT NULL,  -- Range: 0.00-4.00, Mean: 0.01, Median: 0.00
    feeds_u_newsCatInterestsST DECIMAL(10,2) NOT NULL,  -- Range: 0.00-4.00, Mean: 2.30, Median: 3.24
    feeds_u_click_ca2_news DECIMAL(10,2) NOT NULL,  -- Range: 0.00-5.00, Mean: 2.99, Median: 4.68
    feeds_label DECIMAL(10,2) NOT NULL,  -- Range: 0.00-1.00, Mean: 0.10, Median: 0.00
    
    -- Target/Label
    label TINYINT UNSIGNED NOT NULL,  -- Values: 0, 1 (binary classification target)
    
    -- Primary Key
    PRIMARY KEY (id),
    
    -- Indexes for common queries
    INDEX idx_user_id (user_id),
    INDEX idx_label (label),
    INDEX idx_user_label (user_id, label),
    INDEX idx_task_id (task_id),
    INDEX idx_adv_id (adv_id)
    
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
COMMENT='SMOTE synthetic training data - 482,668 rows, balanced 50/50 label distribution';

-- Create synthetic_train_adasyn table (identical structure)
CREATE TABLE synthetic_train_adasyn (
    -- Auto-increment primary key (no natural unique key exists)
    id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    
    -- User Information
    user_id INT UNSIGNED NOT NULL,  -- Range: 100005-148544, 47,466 unique values
    
    -- Demographics
    age TINYINT UNSIGNED NOT NULL,  -- Range: 0-7, 8 unique values
    gender TINYINT UNSIGNED NOT NULL,  -- Range: 0-2, 3 unique values
    residence TINYINT UNSIGNED NOT NULL,  -- Range: 0-34, 35 unique values
    city SMALLINT UNSIGNED NOT NULL,  -- Range: 0-324, 325 unique values
    city_rank TINYINT UNSIGNED NOT NULL,  -- Range: 0-4, 5 unique values
    
    -- Device Information
    series_dev TINYINT UNSIGNED NOT NULL,  -- Range: 0-23, 24 unique values
    series_group TINYINT UNSIGNED NOT NULL,  -- Range: 0-7, 8 unique values
    emui_dev TINYINT UNSIGNED NOT NULL,  -- Range: 0-23, 24 unique values
    device_name TINYINT UNSIGNED NOT NULL,  -- Range: 0-209, 210 unique values
    device_size SMALLINT UNSIGNED NOT NULL,  -- Range: 0-592, 593 unique values
    net_type TINYINT UNSIGNED NOT NULL,  -- Range: 0-6, 7 unique values
    
    -- Advertisement Information
    task_id SMALLINT UNSIGNED NOT NULL,  -- Range: 0-4693, 4,694 unique values
    adv_id SMALLINT UNSIGNED NOT NULL,  -- Range: 0-5205, 5,205 unique values
    creat_type_cd TINYINT UNSIGNED NOT NULL,  -- Range: 0-9, 10 unique values
    adv_prim_id SMALLINT UNSIGNED NOT NULL,  -- Range: 0-408, 409 unique values
    inter_type_cd TINYINT UNSIGNED NOT NULL,  -- Range: 0-3, 4 unique values
    slot_id TINYINT UNSIGNED NOT NULL,  -- Range: 0-54, 55 unique values
    site_id TINYINT UNSIGNED NOT NULL,  -- Range: 0-1, 2 unique values
    spread_app_id TINYINT UNSIGNED NOT NULL,  -- Range: 0-88, 89 unique values
    hispace_app_tags TINYINT UNSIGNED NOT NULL,  -- Range: 0-36, 37 unique values
    app_second_class TINYINT UNSIGNED NOT NULL,  -- Range: 0-20, 21 unique values
    
    -- Floating Point Columns
    app_score DECIMAL(10,2) NOT NULL,  -- Range: 0.00-10.00, Mean: 8.37, Median: 10.00
    u_refreshTimes DECIMAL(10,2) NOT NULL,  -- Range: 0.00-9.00, Mean: 4.86, Median: 5.00
    u_feedLifeCycle DECIMAL(10,2) NOT NULL,  -- Range: 0.00-17.00, Mean: 15.95, Median: 17.00
    
    -- Time Features (all values are -1, indicating missing/unknown)
    hour TINYINT NOT NULL DEFAULT -1,  -- All values: -1 (missing timestamp data)
    dayofweek TINYINT NOT NULL DEFAULT -1,  -- All values: -1 (missing timestamp data)
    
    -- Feeds Domain Features (aggregated user behavior)
    feeds_u_phonePrice DECIMAL(10,2) NOT NULL,  -- Range: 0.00-6.00, Mean: 2.00, Median: 1.00
    feeds_u_browserLifeCycle DECIMAL(10,2) NOT NULL,  -- Range: 0.00-7.00, Mean: 4.13, Median: 6.00
    feeds_u_browserMode DECIMAL(10,2) NOT NULL,  -- Range: 0.00-4.00, Mean: 1.61, Median: 1.80
    feeds_u_feedLifeCycle DECIMAL(10,2) NOT NULL,  -- Range: 0.00-7.00, Mean: 4.02, Median: 5.69
    feeds_u_refreshTimes DECIMAL(10,2) NOT NULL,  -- Range: 0.00-9.00, Mean: 3.67, Median: 3.29
    feeds_u_newsCatInterests DECIMAL(10,2) NOT NULL,  -- Range: 0.00-4.00, Mean: 2.41, Median: 3.77
    feeds_u_newsCatDislike DECIMAL(10,2) NOT NULL,  -- Range: 0.00-4.00, Mean: 0.01, Median: 0.00
    feeds_u_newsCatInterestsST DECIMAL(10,2) NOT NULL,  -- Range: 0.00-4.00, Mean: 2.32, Median: 3.31
    feeds_u_click_ca2_news DECIMAL(10,2) NOT NULL,  -- Range: 0.00-5.00, Mean: 3.02, Median: 4.74
    feeds_label DECIMAL(10,2) NOT NULL,  -- Range: 0.00-1.00, Mean: 0.10, Median: 0.00
    
    -- Target/Label
    label TINYINT UNSIGNED NOT NULL,  -- Values: 0, 1 (binary classification target)
    
    -- Primary Key
    PRIMARY KEY (id),
    
    -- Indexes for common queries
    INDEX idx_user_id (user_id),
    INDEX idx_label (label),
    INDEX idx_user_label (user_id, label),
    INDEX idx_task_id (task_id),
    INDEX idx_adv_id (adv_id)
    
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
COMMENT='ADASYN synthetic training data - 482,815 rows, near-balanced 50/50 label distribution';

-- Sample queries for reference:
-- 
-- 1. Count rows in each table:
--    SELECT COUNT(*) FROM synthetic_train_smote;
--    SELECT COUNT(*) FROM synthetic_train_adasyn;
--
-- 2. Check label distribution:
--    SELECT label, COUNT(*) as count, 
--           ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM synthetic_train_smote), 2) as percentage
--    FROM synthetic_train_smote
--    GROUP BY label;
--
-- 3. Compare statistics between SMOTE and ADASYN:
--    SELECT 'SMOTE' as dataset, AVG(app_score) as avg_app_score, AVG(label) as avg_label
--    FROM synthetic_train_smote
--    UNION ALL
--    SELECT 'ADASYN' as dataset, AVG(app_score) as avg_app_score, AVG(label) as avg_label
--    FROM synthetic_train_adasyn;
--
-- 4. Find users with most records:
--    SELECT user_id, COUNT(*) as record_count
--    FROM synthetic_train_smote
--    GROUP BY user_id
--    ORDER BY record_count DESC
--    LIMIT 10;
--
-- 5. Query by label:
--    SELECT * FROM synthetic_train_smote WHERE label = 1 LIMIT 10;
--    SELECT * FROM synthetic_train_adasyn WHERE label = 0 LIMIT 10;

