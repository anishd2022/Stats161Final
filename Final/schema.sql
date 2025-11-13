-- MySQL Schema for Ads and Feeds Tables
-- Based on schema exploration of ads_mini.csv and feeds_mini.csv

-- Drop tables if they exist (for clean setup)
DROP TABLE IF EXISTS feeds;
DROP TABLE IF EXISTS ads;

-- Create ads table
CREATE TABLE ads (
    -- Primary Key
    log_id INT UNSIGNED NOT NULL,
    
    -- Target/Label
    label TINYINT UNSIGNED NOT NULL,  -- Values: 0, 1
    
    -- User Information
    user_id INT UNSIGNED NOT NULL,  -- Range: 100005-107676, Join key with feeds
    
    -- Demographics
    age TINYINT UNSIGNED NOT NULL,  -- Range: 2-9
    gender TINYINT UNSIGNED NOT NULL,  -- Values: 2, 3, 4
    residence TINYINT UNSIGNED NOT NULL,  -- Range: 11-46
    city SMALLINT UNSIGNED NOT NULL,  -- Range: 101-441
    city_rank TINYINT UNSIGNED NOT NULL,  -- Values: 2, 3, 4, 5
    
    -- Device Information
    series_dev TINYINT UNSIGNED NOT NULL,  -- Range: 11-36
    series_group TINYINT UNSIGNED NOT NULL,  -- Range: 2-8
    emui_dev TINYINT UNSIGNED NOT NULL,  -- Range: 11-37
    device_name SMALLINT UNSIGNED NOT NULL,  -- Range: 101-357
    device_size SMALLINT UNSIGNED NOT NULL,  -- Range: 1008-2579
    net_type TINYINT UNSIGNED NOT NULL,  -- Values: 2, 3, 4, 5, 6, 7
    
    -- Advertisement Information
    task_id SMALLINT UNSIGNED NOT NULL,  -- Range: 10025-36344
    adv_id SMALLINT UNSIGNED NOT NULL,  -- Range: 10004-23579
    creat_type_cd TINYINT UNSIGNED NOT NULL,  -- Range: 2-10
    adv_prim_id SMALLINT UNSIGNED NOT NULL,  -- Range: 1005-2077
    inter_type_cd TINYINT UNSIGNED NOT NULL,  -- Values: 3, 4, 5
    slot_id TINYINT UNSIGNED NOT NULL,  -- Range: 12-71
    site_id TINYINT UNSIGNED NOT NULL DEFAULT 1,  -- Always 1
    spread_app_id SMALLINT UNSIGNED NOT NULL,  -- Range: 101-372
    hispace_app_tags TINYINT UNSIGNED NOT NULL,  -- Range: 12-52
    app_second_class TINYINT UNSIGNED NOT NULL,  -- Range: 11-30
    app_score DECIMAL(3,1) NOT NULL,  -- Values: 0.0, 6.0, 10.0
    
    -- List columns stored as JSON arrays
    -- ad_click_list_v001: Array of integers (typically 4 elements)
    ad_click_list_v001 JSON NOT NULL,
    -- ad_click_list_v002: Array of integers (typically 3 elements)
    ad_click_list_v002 JSON NOT NULL,
    -- ad_click_list_v003: Array of integers (typically 3 elements)
    ad_click_list_v003 JSON NOT NULL,
    -- ad_close_list_v001: Array of strings (typically 1 element)
    ad_close_list_v001 JSON NOT NULL,
    -- ad_close_list_v002: Array of strings (typically 1 element)
    ad_close_list_v002 JSON NOT NULL,
    -- ad_close_list_v003: Array of strings (typically 1 element)
    ad_close_list_v003 JSON NOT NULL,
    
    -- Timestamp
    pt_d BIGINT UNSIGNED NOT NULL,  -- Range: 202206020341-202206041209 (timestamp format)
    
    -- User Interest/Behavior Lists
    -- u_newsCatInterestsST: Array of integers (typically 3 elements)
    u_newsCatInterestsST JSON NOT NULL,
    u_refreshTimes TINYINT UNSIGNED NOT NULL,  -- Range: 0-9
    u_feedLifeCycle TINYINT UNSIGNED NOT NULL,  -- Range: 10-17
    
    -- Primary Key
    PRIMARY KEY (log_id),
    
    -- Index on user_id for joins with feeds table
    INDEX idx_user_id (user_id),
    
    -- Index on label for filtering
    INDEX idx_label (label),
    
    -- Index on timestamp for time-based queries
    INDEX idx_pt_d (pt_d)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Create feeds table
CREATE TABLE feeds (
    -- User Information (Join key with ads)
    u_userId INT UNSIGNED NOT NULL,  -- Range: 100022-287169, Join key with ads
    
    -- User Device/Behavior
    u_phonePrice TINYINT UNSIGNED NOT NULL,  -- Range: 10-16
    u_browserLifeCycle TINYINT UNSIGNED NOT NULL,  -- Range: 10-17
    u_browserMode TINYINT UNSIGNED NOT NULL,  -- Range: 10-16
    u_feedLifeCycle TINYINT UNSIGNED NOT NULL,  -- Range: 10-17
    u_refreshTimes TINYINT UNSIGNED NOT NULL,  -- Range: 0-9
    
    -- User Interest Lists (stored as JSON arrays)
    -- u_newsCatInterests: Array of integers (typically 5 elements)
    u_newsCatInterests JSON NOT NULL,
    -- u_newsCatDislike: Array of strings (typically 1 element)
    u_newsCatDislike JSON NOT NULL,
    -- u_newsCatInterestsST: Array of integers (typically 5 elements)
    u_newsCatInterestsST JSON NOT NULL,
    -- u_click_ca2_news: Array of integers (typically 5 elements)
    u_click_ca2_news JSON NOT NULL,
    
    -- Item/Content Information
    i_docId VARCHAR(40) NOT NULL,  -- Hash string, max length 40
    i_s_sourceId VARCHAR(40) NOT NULL,  -- Hash string, max length 40
    i_regionEntity SMALLINT UNSIGNED NOT NULL,  -- Range: 0-3187
    i_cat TINYINT UNSIGNED NOT NULL,  -- Range: 0-220
    -- i_entities: Array of hash strings (typically 5 elements, each 64 chars)
    i_entities JSON NOT NULL,
    i_dislikeTimes TINYINT UNSIGNED NOT NULL,  -- Range: 0-9
    i_upTimes TINYINT UNSIGNED NOT NULL,  -- Range: 0-9
    i_dtype TINYINT UNSIGNED NOT NULL,  -- Values: 10, 11, 13, 14
    
    -- Event/Engagement Information
    e_ch TINYINT UNSIGNED NOT NULL,  -- Range: 1-20
    e_m SMALLINT UNSIGNED NOT NULL,  -- Range: 14-1483
    e_po TINYINT UNSIGNED NOT NULL,  -- Range: 1-27
    e_pl SMALLINT UNSIGNED NOT NULL,  -- Range: 0-3189
    e_rn TINYINT UNSIGNED NOT NULL,  -- Range: 1-59
    e_section TINYINT UNSIGNED NOT NULL,  -- Values: 0, 1
    e_et BIGINT UNSIGNED NOT NULL,  -- Range: 202206080010-202206082329 (timestamp format)
    
    -- Labels/Targets
    label TINYINT NOT NULL,  -- Values: -1, 1
    cillabel TINYINT NOT NULL,  -- Values: -1, 1
    pro TINYINT UNSIGNED NOT NULL,  -- Values: 0, 10, 20, 40, 60, 80, 100
    
    -- Note: No single-column primary key identified in feeds table
    -- Using composite key or auto-increment ID would be needed for production
    -- For now, we'll create an auto-increment primary key
    id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    
    -- Primary Key (auto-increment)
    PRIMARY KEY (id),
    
    -- Index on user_id for joins with ads table
    INDEX idx_u_userId (u_userId),
    
    -- Index on labels for filtering
    INDEX idx_label (label),
    INDEX idx_cillabel (cillabel),
    
    -- Index on timestamp for time-based queries
    INDEX idx_e_et (e_et),
    
    -- Index on document ID for content lookups
    INDEX idx_i_docId (i_docId),
    
    -- Composite index for common queries (user + timestamp)
    INDEX idx_user_timestamp (u_userId, e_et)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Create a view for joined data (optional, for convenience)
CREATE OR REPLACE VIEW ads_feeds_joined AS
SELECT 
    a.log_id,
    a.label AS ads_label,
    a.user_id,
    a.age,
    a.gender,
    a.residence,
    a.city,
    a.pt_d AS ad_timestamp,
    f.id AS feed_id,
    f.u_userId,
    f.label AS feed_label,
    f.cillabel,
    f.pro,
    f.e_et AS feed_timestamp,
    f.i_docId,
    f.i_cat
FROM ads a
LEFT JOIN feeds f ON a.user_id = f.u_userId;

-- Add comments for documentation
ALTER TABLE ads COMMENT = 'Advertisement interaction data with user demographics and device information';
ALTER TABLE feeds COMMENT = 'Feed content interaction data with user behavior and content features';

-- Sample queries for reference:
-- 
-- 1. Join ads and feeds on user_id:
--    SELECT * FROM ads a
--    INNER JOIN feeds f ON a.user_id = f.u_userId;
--
-- 2. Query JSON array columns (example for ad_click_list_v001):
--    SELECT log_id, JSON_EXTRACT(ad_click_list_v001, '$[0]') AS first_click_id
--    FROM ads;
--
-- 3. Filter by label:
--    SELECT * FROM ads WHERE label = 1;
--    SELECT * FROM feeds WHERE label = 1;
--
-- 4. Count users in both tables:
--    SELECT COUNT(DISTINCT a.user_id) AS users_in_both
--    FROM ads a
--    INNER JOIN feeds f ON a.user_id = f.u_userId;

