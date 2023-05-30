#include<stdio.h>
int prime_number(int l,int h)
{
    int i;
    for(i=0;i<h;i++)
    {
        if(h%i==0 && l<h)
        {
            break;
        }
        else
        {
            printf("%d",i);

        }

    }
}
int main()
{
    int l,h;
    scanf("%d",l);
    scanf("%d",h);
    prime_number(l,h);
}