from django.db.models import Aggregate, CharField


class MySQLGroupConcatMixin:
    def as_mysql(self, compiler, connection, separator=',', **extra_content):
        return super().as_sql(compiler,
                              connection,
                              template='%(function)s(%(distinct)s%(expressions)s%(ordering)s%(separator)s)',
                              separator=' SEPARATOR \'%s\'' % separator)


class GroupConcat(Aggregate, MySQLGroupConcatMixin):
    '''
    For the more details @see https://stackoverflow.com/questions/10340684/group-concat-equivalent-in-django
    The @WeizhongTu answer and @Jboulery (addition)
    '''

    function = 'GROUP_CONCAT'
    separator = ','

    def __init__(self, expression, distinct=False, ordering=None, **extra):
        super(GroupConcat, self).__init__(expression,
                                          distinct='DISTINCT ' if distinct else '',
                                          ordering=' ORDER BY %s' % ordering if ordering is not None else '',
                                          output_field=CharField(),
                                          **extra)

    def as_sql(self, compiler, connection, **extra):
        return super().as_sql(compiler,
                              connection,
                              template='%(function)s(%(distinct)s%(expressions)s%(ordering)s)',
                              **extra)



class OracleGroupConcatMixin:
    def as_oracle(self, compiler, connection, **extra_context):
        # return super().as_sql(
        #     compiler,
        #     connection,
        #     template=(
        #         "LOWER(RAWTOHEX(STANDARD_HASH(UTL_I18N.STRING_TO_RAW("
        #         "%(expressions)s, 'AL32UTF8'), '%(function)s')))"
        #     ),
        #     **extra_context,
        # )
        ...


class PostgreSQLSHAMixin:
    def as_postgresql(self, compiler, connection, **extra_content):
        # return super().as_sql(
        #     compiler,
        #     connection,
        #     template="ENCODE(DIGEST(%(expressions)s, '%(function)s'), 'hex')",
        #     function=self.function.lower(),
        #     **extra_content,
        # )
        ...


'''
For postgres:
Select name_of_column1, name_of_column2, name_of_column3, ….., name_of_columnN array_to_string (array_agg (name_of_column), “Value separator”) from name_of_table JOIN condition Group by condition;

SELECT id_field, array_agg(value_field1), array_agg(value_field2)
FROM data_table
GROUP BY id_field


For oracle:
Select First_column,LISTAGG(second_column,',') 
    WITHIN GROUP (ORDER BY second_column) as Sec_column, 
    LISTAGG(third_column,',') 
    WITHIN GROUP (ORDER BY second_column) as thrd_column 
FROM tablename 
GROUP BY first_column


SELECT STRING_AGG(Genre, ',') AS Result
FROM Genres;

https://docs.djangoproject.com/en/4.1/ref/models/database-functions/
https://docs.djangoproject.com/en/dev/ref/models/database-functions/


'''
