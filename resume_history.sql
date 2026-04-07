IF OBJECT_ID('dbo.resume_history', 'U') IS NOT NULL
    DROP TABLE dbo.resume_history;
GO

CREATE TABLE dbo.resume_history (
    id INT IDENTITY(1,1) PRIMARY KEY,
    user_id INT NULL,
    score FLOAT NULL,
    filename NVARCHAR(255) NULL,
    filetype NVARCHAR(50) NULL,
    [date] NVARCHAR(30) NULL,
    found NVARCHAR(MAX) NULL,
    missing NVARCHAR(MAX) NULL,
    rewritten NVARCHAR(MAX) NULL,
    skills NVARCHAR(MAX) NULL,
    batch_id NVARCHAR(80) NULL,
    folder_name NVARCHAR(255) NULL
);
